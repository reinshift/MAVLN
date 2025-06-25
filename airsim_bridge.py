"""
when env_airsim simulator is started, this bridge script will play a role as Server, 
another script as Client to catch actions that UAVs have taken in `main.py`, 
and control UAVs to take the corresponding actions in simulator by sending commands to the Server.
"""
import airsim
import threading
import logging
import socket
import json
import time
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import io
import base64
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AirSimBridge:
    def __init__(self, host="localhost", port=8888, sim_path=None, config_path=None):
        """
        Args:
            host: host of the Server
            port: port of the Server
            sim_path: path to the simulator, use the default path if not provided
            config_path: path to the config file
        """
        self.host = host
        self.port = port
        self.config_path = config_path

        if sim_path is None:
            default_paths = [
                os.path.join(os.path.dirname(__file__), "envs/airsim/env_airsim_23/LinuxNoEditor/start.sh"),
                "envs/airsim/env_airsim_23/LinuxNoEditor/start.sh"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    sim_path = path
                    break
                
        self.sim_path = sim_path
        
        if self.sim_path is None or not os.path.exists(self.sim_path):
            logger.warning(f"Simulator path not found. Please provide a valid path or check default locations.")
            self.sim_path = None
        else:
            logger.info(f"Using simulator path: {self.sim_path}")
        
        self.server_socket = None
        self.client_socket = {} # {uav_id: socket}
        self.sim_process = None
        self.sim_thread = None
        self.server_thread = None
        self.running = False
        
        # connect to Airsim
        self.client = None

        # create a dict to control UAVs
        self.uav_controllers = {}

    def start_simulator(self):
        if self.sim_path is None:
            logger.error("Simulator path not set, cannot start simulator")
            return False
            
        logger.info(f"Starting simulator: {self.sim_path}")
        try:
            self.sim_process = subprocess.Popen(self.sim_path, shell=True)
            logger.info(f"Simulator started successfully, pid {self.sim_process.pid}")

            time.sleep(5)

            self.connect_to_airsim()
            return True
        except Exception as e:
            logger.error(f"Failed to start simulator: {str(e)}")
            return False
        
    def connect_to_airsim(self):
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            logger.info("Connected to Airsim successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Airsim: {str(e)}")
            return False
    
    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"listening on {self.host}:{self.port}")

            self.running = True
            self.server_thread = threading.Thread(target=self._handle_connections)
            self.server_thread.daemon = True
            self.server_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            return False

    def _handle_connections(self):
        """function to handle threads connected to client"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"client {addr} connected")

                # create thread
                client_thread = threading.Thread(target=self._handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Error in connecting to client: {str(e)}")
    
    def _handle_client(self, client_socket, addr):
        uav_id = None
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                return

            # parse the message from client
            try:
                msg = json.loads(data)
                if msg.get("type") == "init":
                    uav_id = msg.get("uav_id")
                    if uav_id:
                        self.client_socket[uav_id] = client_socket
                        logger.info(f"UAV {uav_id} registered")

                        self._init_uav_controller(uav_id)

                        response = {
                            'status': 'ok',
                            'message': f'UAV {uav_id} initialized'
                        }
                        client_socket.sendall(json.dumps(response).encode('utf-8'))
                    
                        while self.running:
                            data = client_socket.recv(4096).decode('utf-8')
                            if not data:
                                break

                            self._process_client_message(uav_id, data, client_socket)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format {data}")
        except Exception as e:
            logger.error(f"Error in handling client {addr}: {str(e)}")
        finally:
            # client disconnected
            try:
                if uav_id in self.client_socket:
                    del self.client_socket[uav_id]
                    client_socket.close()
                    logger.info(f"client disconnected")
            except:
                pass

    def _process_client_message(self, uav_id, data, client_socket):
        """process messages from client"""
        try:
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == 'get_image':
                image = self.get_uav_image(uav_id)
                if image:
                    # image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    response = {
                        'status': 'ok',
                        'type': 'image',
                        'image': img_str
                    }
                else:
                    response = {
                        'status': 'error',
                        'message': 'Failed to get image'
                    }
                client_socket.sendall(json.dumps(response).encode('utf-8'))
            
            elif msg_type == 'execute_action':
                action = msg.get("action")
                result = self.execute_uav_action(uav_id, action)

                response = {
                    'status': 'ok' if result else 'error',
                    'message': f'execute action {action} {"successfully" if result else "failed"}'
                }
                client_socket.sendall(json.dumps(response).encode('utf-8'))
        
            elif msg_type == 'set_position':
                position = msg.get("position")
                if position and len(position) == 6:
                    result = self.set_uav_position(uav_id, position)
                    response = {
                        'status': 'ok' if result else 'error',
                        'message': f'set position {position} {"successfully" if result else "failed"}'
                    }
                else:
                    response = {
                        'status': 'error',
                        'message': 'Invalid position format'
                    }
                client_socket.sendall(json.dumps(response).encode('utf-8'))

            else:
                response = {
                    'status': 'error',
                    'message': f'Invalid message type: {msg_type}'
                }
                client_socket.sendall(json.dumps(response).encode('utf-8'))
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format {data}")
        except Exception as e:
            logger.error(f"Error in processing message from client {str(e)}")

    def _init_uav_controller(self, uav_id):
        try:
            if not self.client:
                self.connect_to_airsim()
            
            # create controller based on uav_id
            vehicle_name = f"UAV_{uav_id}"
            
            # enable API control
            self.client.enableApiControl(True, vehicle_name)
            
            # unlock
            self.client.armDisarm(True, vehicle_name)
            
            # store controller info
            self.uav_controllers[uav_id] = {
                'vehicle_name': vehicle_name,
                'initialized': True
            }
            
            logger.info(f"UAV {uav_id} controller initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize UAV {uav_id} controller: {str(e)}")
            return False

    def get_uav_id(self):
        """get all registered UAV IDs"""
        return list(self.uav_controllers.keys())

    def set_uav_position(self, uav_id, position):
        """
        Args:
            uav_id: UAV ID
            position: [x, y, z, yaw, pitch, roll]
        """
        try:
            if uav_id not in self.uav_controllers:
                logger.error(f"UAV {uav_id} not initialized")
                return False
            vehicle_name = self.uav_controllers[uav_id]['vehicle_name']
            position = airsim.Pose(position[0], position[1], position[2], 
                                   airsim.to_quaternion(0, 0, position[5]))
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
            self.client.simSetVehiclePose(position, True, vehicle_name)
            logger.info(f"UAV {uav_id} position set to {position}")
            return True
        except Exception as e:
            logger.error(f"Failed to set position for UAV {uav_id}: {str(e)}")
            return False

    def get_uav_image(self, uav_id):
        """
        Args:
            uav_id: UAV ID
        Returns:
            image: PIL Image object
        """
        try:
            if uav_id not in self.uav_controllers:
                logger.error(f"UAV {uav_id} not initialized")
                return None
            
            vehicle_name = self.uav_controllers[uav_id]['vehicle_name']
            
            # get image
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_custom", airsim.ImageType.Scene, False, False)
            ], vehicle_name)
            
            if responses:
                # convert image data to PIL.Image
                img_data = responses[0]
                img1d = np.fromstring(img_data.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(img_data.height, img_data.width, 3)
                img = Image.fromarray(img_rgb)
                return img
            else:
                logger.error(f"Failed to get image for UAV {uav_id}: no response")
                return None
        except Exception as e:
            logger.error(f"Failed to get image for UAV {uav_id}: {str(e)}")
            return None

    def execute_uav_action(self, uav_id, action):
        """
        Args:
            uav_id: UAV ID
            action: action to execute
        Returns:
            result: True if action executed successfully, False otherwise
        """
        try:
            if uav_id not in self.uav_controllers:
                logger.error(f"UAV {uav_id} 未初始化")
                return False
            
            vehicle_name = self.uav_controllers[uav_id]['vehicle_name']
            
            """action_map:
                0: "stop"
                1: "go straight"
                2: "turn left"
                3: "turn right"
                4: "go up"
                5: "go down"
                6: "move left"
                7: "move right" """
            
            if action == 0:  # stop
                self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=vehicle_name)
            elif action == 1:  # go straight
                self.client.moveByVelocityAsync(1, 0, 0, 1, vehicle_name=vehicle_name)
            elif action == 2:  # turn left
                self.client.rotateByYawRateAsync(-30, 1, vehicle_name=vehicle_name)
            elif action == 3:  # turn right
                self.client.rotateByYawRateAsync(30, 1, vehicle_name=vehicle_name)
            elif action == 4:  # go up
                self.client.moveByVelocityAsync(0, 0, -1, 1, vehicle_name=vehicle_name)
            elif action == 5:  # go down
                self.client.moveByVelocityAsync(0, 0, 1, 1, vehicle_name=vehicle_name)
            elif action == 6:  # move left
                self.client.moveByVelocityAsync(0, -1, 0, 1, vehicle_name=vehicle_name)
            elif action == 7:  # move right
                self.client.moveByVelocityAsync(0, 1, 0, 1, vehicle_name=vehicle_name)  
            else:
                logger.warning(f"Unknown action ID: {action}, execute stop action")
                self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=vehicle_name)
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to execute action {action} for UAV {uav_id}: {str(e)}")
            return False
    
    def start(self):
        """start AirSimBridge"""
        if not self.start_simulator():
            return False
        
        # start TCP server
        if not self.start_server():
            return False
        
        logger.info("AirSimBridge started successfully")
        return True
    
    def stop(self):
        """stop AirSimBridge"""
        self.running = False
        
        # close all client connections
        for client_socket in self.client_socket.values():  
            try:
                client_socket.close()
            except:
                pass
        
        # close server
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # stop simulator
        if self.sim_process:
            try:
                self.sim_process.terminate()
                self.sim_process.wait()
            except:
                pass
        
        logger.info("AirSimBridge stopped")

def main():
    parser = argparse.ArgumentParser(description='AirSim Bridge Server')
    parser.add_argument('--host', type=str, default='localhost', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on')
    parser.add_argument('--sim-path', type=str, default=None, help='Path to AirSim simulator')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    args = parser.parse_args()
    
    bridge = AirSimBridge(args.host, args.port, args.sim_path, args.config)
    
    if bridge.start():
        try:
            # keep main thread running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        finally:
            bridge.stop()
    else:
        logger.error("AirSimBridge started failed")

if __name__ == "__main__":
    main()