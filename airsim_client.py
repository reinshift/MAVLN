"""
when env_airsim simulator is started, this script will play a role as Client, 
get images from the Server, and feed them to the model with instructions to take actions.
finally, send the actions to the Server.
"""
import airsim
import logging
import socket
import json
import base64
import io
import time
import numpy as np
from PIL import Image
import sys
import os
import threading
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mavln.mavln import MAVLN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AirSimClient:
    def __init__(self, config_path, host="localhost", port=8888, update_frequency=1.0):
        """
        initialize the AirSim client
        
        Args:
            config_path: path to the config file
            host: host name of the AirSim Bridge server
            port: port of the AirSim Bridge server
            update_frequency: update frequency(Hz)
        """
        self.host = host
        self.port = port
        self.update_frequency = update_frequency
        self.running = False
        self.sockets = {}  # {uav_id: socket}
        self.update_thread = None
        
        # load the config file
        self.config = self._load_config(config_path)
        
        # initialize the MAVLN model
        logger.info("Initializing MAVLN model...")
        self.model = MAVLN(self.config)
        logger.info("MAVLN model initialized")
        
        # number of UAVs
        self.num_uavs = self.config.data.num_agents
        logger.info(f"Number of UAVs: {self.num_uavs}")
        
        # action mapping
        self.action_map = self.config.model.action_map
        logger.info(f"Action mapping: {self.action_map}")
        
        # initialize the instructions
        self.instructions = ["move forward along the road, stop at the front of building" for _ in range(self.num_uavs)]

    def _load_config(self, config_path):
        """load the config file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        class Config:
            def __init__(self, d):
                self._dict_props = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        if all(isinstance(key, int) for key in v.keys()):
                            self._dict_props[k] = v
                        else:
                            setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)
                    
            def __getattr__(self, name):
                if name in self._dict_props:
                    return self._dict_props[name]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        return Config(config_dict)

    def connect(self):
        """connect to the AirSim Bridge server and initialize all UAVs"""
        logger.info(f"Connecting to the AirSim Bridge server {self.host}:{self.port}...")
        
        for uav_id in range(1, self.num_uavs + 1):
            try:
                # create a socket connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                
                # send the initialization message
                init_msg = {
                    "type": "init",
                    "uav_id": uav_id
                }
                sock.sendall(json.dumps(init_msg).encode('utf-8'))
                
                # receive the response
                response = sock.recv(1024).decode('utf-8')
                response_data = json.loads(response)
                
                if response_data.get('status') == 'ok':
                    logger.info(f"UAV {uav_id} initialized successfully")
                    self.sockets[uav_id] = sock
                else:
                    logger.error(f"UAV {uav_id} initialization failed: {response_data.get('message')}")
                    sock.close()
            except Exception as e:
                logger.error(f"Failed to connect to UAV {uav_id}: {str(e)}")
        
        if len(self.sockets) == 0:
            logger.error("No UAVs connected successfully")
            return False
        
        logger.info(f"Successfully connected {len(self.sockets)} UAVs")
        return True

    def get_image(self, uav_id):
        """get the image from the UAV"""
        if uav_id not in self.sockets:
            logger.error(f"UAV {uav_id} is not connected")
            return None
        
        try:
            sock = self.sockets[uav_id]
            
            # send the request to get the image
            msg = {
                "type": "get_image"
            }
            sock.sendall(json.dumps(msg).encode('utf-8'))
            
            # receive the image data
            data = sock.recv(1024 * 1024).decode('utf-8')  # increase the buffer size to receive large images
            response_data = json.loads(data)
            
            if response_data.get('status') == 'ok' and 'image' in response_data:
                # decode the base64 image
                img_data = base64.b64decode(response_data['image'])
                img = Image.open(io.BytesIO(img_data))
                return img
            else:
                logger.error(f"Failed to get the image from UAV {uav_id}: {response_data.get('message', 'unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Failed to get the image from UAV {uav_id}: {str(e)}")
            return None

    def execute_action(self, uav_id, action):
        """execute the action of the UAV"""
        if uav_id not in self.sockets:
            logger.error(f"UAV {uav_id} is not connected")
            return False
        
        try:
            sock = self.sockets[uav_id]
            
            # send the request to execute the action
            msg = {
                "type": "execute_action",
                "action": action
            }
            sock.sendall(json.dumps(msg).encode('utf-8'))
            
            # receive the response
            response = sock.recv(1024).decode('utf-8')
            response_data = json.loads(response)
            
            if response_data.get('status') == 'ok':
                action_name = self.action_map.get(action, f"unknown action {action}")
                logger.info(f"UAV {uav_id} executed action {action} ({action_name}) successfully")
                return True
            else:
                logger.error(f"UAV {uav_id} executed action {action} failed: {response_data.get('message', 'unknown error')}")
                return False
        except Exception as e:
            logger.error(f"UAV {uav_id} executed action {action} failed: {str(e)}")
            return False

    def set_uav_position(self, uav_id, position):
        """set the position of the UAV"""
        if uav_id not in self.sockets:
            logger.error(f"UAV {uav_id} is not connected")
            return False
        
        try:
            sock = self.sockets[uav_id]
            
            # send the request to set the position
            msg = {
                "type": "set_position",
                "position": position  # [x, y, z, yaw, pitch, roll]
            }
            sock.sendall(json.dumps(msg).encode('utf-8'))
            
            # receive the response
            response = sock.recv(1024).decode('utf-8')
            response_data = json.loads(response)
            
            if response_data.get('status') == 'ok':
                logger.info(f"UAV {uav_id} set position {position} successfully")
                return True
            else:
                logger.error(f"UAV {uav_id} set position {position} failed: {response_data.get('message', 'unknown error')}")
                return False
        except Exception as e:
            logger.error(f"UAV {uav_id} set position {position} failed: {str(e)}")
            return False

    def set_instruction(self, uav_id, instruction):
        """set the instruction of the UAV"""
        if 1 <= uav_id <= self.num_uavs:
            self.instructions[uav_id - 1] = instruction
            logger.info(f"UAV {uav_id} set instruction: {instruction}")
            return True
        else:
            logger.error(f"Invalid UAV ID: {uav_id}")
            return False

    def update_loop(self):
        """update loop, get images, call model, send actions"""
        logger.info("Starting update loop")
        
        while self.running:
            start_time = time.time()
            
            try:
                # collect images from all UAVs
                agent_images = []
                for uav_id in range(1, self.num_uavs + 1):
                    if uav_id in self.sockets:
                        img = self.get_image(uav_id)
                        if img:
                            agent_images.append([img])
                        else:
                            # if cannot get image, use blank image instead
                            blank_img = Image.new('RGB', (224, 224), (255, 255, 255))
                            agent_images.append([blank_img])
                            logger.warning(f"Cannot get image from UAV {uav_id}, use blank image instead")
                    else:
                        # if UAV is not connected, use blank image instead
                        blank_img = Image.new('RGB', (224, 224), (255, 255, 255))
                        agent_images.append([blank_img])
                        logger.warning(f"UAV {uav_id} is not connected, use blank image instead")
                
                # ensure the number of images matches the number of UAVs
                if len(agent_images) != self.num_uavs:
                    logger.warning(f"Number of images ({len(agent_images)}) does not match the number of UAVs ({self.num_uavs})")
                    if len(agent_images) < self.num_uavs:
                        # if images are not enough, add blank images
                        for _ in range(self.num_uavs - len(agent_images)):
                            blank_img = Image.new('RGB', (224, 224), (255, 255, 255))
                            agent_images.append([blank_img])
                    else:
                        # if images are too many, truncate
                        agent_images = agent_images[:self.num_uavs]
                
                # call the MAVLN model for inference
                logger.info("Calling MAVLN model for inference...")
                response = self.model(agent_images, self.instructions)
                logger.info(f"Model response: {response}")
                
                # parse the actions
                actions = self.model.parse_actions(response)
                logger.info(f"Parsed actions: {actions}")
                
                # execute the actions
                for agent_idx, action in actions.items():
                    uav_id = agent_idx + 1  # convert from 0-based index to 1-based ID
                    if uav_id in self.sockets:
                        self.execute_action(uav_id, action)
                    else:
                        logger.warning(f"UAV {uav_id} is not connected, cannot execute action")
            
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
            
            # control the update frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / self.update_frequency - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Update loop took longer than expected: {elapsed:.2f} seconds, exceeded expected period {1.0/self.update_frequency:.2f} seconds")

    def start(self):
        """start the client"""
        if self.running:
            logger.warning("Client is already running")
            return False
        
        # connect to the AirSim Bridge server
        if not self.connect():
            logger.error("Failed to connect to the AirSim Bridge server")
            return False
        
        # start the update loop
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Client started")
        return True

    def stop(self):
        """stop the client"""
        self.running = False
        
        # wait for the update thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        
        # close all socket connections
        for uav_id, sock in self.sockets.items():
            try:
                sock.close()
                logger.info(f"Closed connection to UAV {uav_id}")
            except:
                pass
        
        self.sockets.clear()
        logger.info("Client stopped")

def main():
    parser = argparse.ArgumentParser(description='AirSim client')
    parser.add_argument('--config', type=str, default='configs/common.yaml', help='path to the config file')
    parser.add_argument('--host', type=str, default='localhost', help='host name of the AirSim Bridge server')
    parser.add_argument('--port', type=int, default=8888, help='port of the AirSim Bridge server')
    parser.add_argument('--freq', type=float, default=1.0, help='update frequency(Hz)')
    args = parser.parse_args()
    
    client = AirSimClient(
        config_path=args.config,
        host=args.host,
        port=args.port,
        update_frequency=args.freq
    )
    
    if client.start():
        try:
            # keep the main thread running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        finally:
            client.stop()
    else:
        logger.error("Client startup failed")

if __name__ == "__main__":
    main()

