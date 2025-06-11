import os
import sys
import torch
import logging
import argparse
from PIL import Image
import yaml
import matplotlib.pyplot as plt

"""
test_mavln.py is used to test end-to-end performance of MAVLN model.
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mavln.mavln import MAVLN
from utils.parquet_reader import ParquetReader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
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

def visualize_agent_images(images, actions, instructions=None, actual_actions=None, save_path=None):
    """Visualize agent images and actions."""
    num_agents = len(images)
    fig, axes = plt.subplots(1, num_agents, figsize=(num_agents * 4, 4))
    
    if num_agents == 1:
        axes = [axes]
    
    for i, (img, action) in enumerate(zip(images, actions.items())):
        axes[i].imshow(img)
        title = f"Agent {i+1}: Action {action[1]}"
        
        # Add actual action information
        if actual_actions and i in actual_actions:
            title += f"\nActual: {actual_actions[i]}"
            
        if instructions and i < len(instructions):
            instr_short = instructions[i][:20] + "..." if len(instructions[i]) > 20 else instructions[i]
            title += f"\n{instr_short}"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def test_multi_agent_trajectories(model, dataset, traj_ids, save_dir=None):
    """Test multi-agent trajectories."""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Select trajectory and frame for each agent
    agent_images = []
    agent_instructions = []
    actual_actions = {}  # Store actual actions

    if len(traj_ids) < model.num_agents:
        logger.warning(f"Trajectory number {len(traj_ids)} less than agent number {model.num_agents}, will repeat trajectories")
        # Repeat trajectory IDs to match agent number
        traj_ids = traj_ids * ((model.num_agents // len(traj_ids)) + 1)
        traj_ids = traj_ids[:model.num_agents]
    
    used_traj_ids = []
    for i in range(model.num_agents):
        traj_id = traj_ids[i]
        used_traj_ids.append(traj_id)
        
        traj_data = dataset.get_trajectory(traj_id)
        
        if not traj_data or len(traj_data) == 0:
            logger.warning(f"Trajectory {traj_id} has no data, use default image")
            if agent_images:
                agent_images.append([agent_images[0][0]])
            else:
                blank_img = Image.new('RGB', (224, 224), (255, 255, 255))
                agent_images.append([blank_img])
            agent_instructions.append("No available instruction")
            actual_actions[i] = -1  # Invalid action
            continue
        
        instruction = traj_data[0]['instruction'] if 'instruction' in traj_data[0] else f"智能体{i+1}的默认指令"
        agent_instructions.append(instruction)
        
        img_tensor = traj_data[0]['image']
        
        actual_actions[i] = traj_data[0]['action_value'].item() if isinstance(traj_data[0]['action_value'], torch.Tensor) else traj_data[0]['action_value']
        
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = img * torch.tensor([0.229, 0.224, 0.225]).numpy() + torch.tensor([0.485, 0.456, 0.406]).numpy()
            img = (img * 255).clip(0, 255).astype('uint8')
            pil_img = Image.fromarray(img)
        else:
            pil_img = img_tensor
        
        agent_images.append([pil_img])
    
    logger.info(f"Using {len(agent_images)} different trajectories for model inference...")
    logger.info(f"Used trajectory IDs: {used_traj_ids}")
    
    response = model(agent_images, agent_instructions)
    logger.info(f"Model response: {response}")
    
    # Parse actions
    actions = model.parse_actions(response)
    logger.info(f"Parsed actions: {actions}")
    logger.info(f"Actual actions: {actual_actions}")
    
    # Get first image for each agent for visualization
    viz_images = [imgs[0] for imgs in agent_images]
    
    # Visualize results
    if save_dir:
        save_path = os.path.join(save_dir, f"multi_agent_visualization.png")
        visualize_agent_images(viz_images, actions, agent_instructions, actual_actions, save_path)
    
    return response, actions, agent_instructions, used_traj_ids, actual_actions

def main():
    parser = argparse.ArgumentParser(description='Test MAVLN model')
    parser.add_argument('--config', type=str, default='../configs/common.yaml', help='Configuration file path')
    parser.add_argument('--save_dir', type=str, default='../results', help='Save directory')
    parser.add_argument('--num_tests', type=int, default=3, help='Test times')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_reader = ParquetReader(config)
    val_loader = data_reader.read_parquet(split='train')
    
    # Initialize model
    logger.info("Initializing MAVLN model...")
    model = MAVLN(config)
    
    # Get trajectory IDs from dataset
    dataset = val_loader.dataset
    traj_ids = dataset.metadata_df['traj_id'].unique().tolist()
    
    if not traj_ids:
        logger.error("No trajectories found in dataset")
        return
    
    # Execute multiple tests, each using different trajectory combinations
    for test_idx in range(args.num_tests):
        logger.info(f"Executing test {test_idx+1}/{args.num_tests}")
        
        # Select trajectory for this test
        start_idx = (test_idx * model.num_agents) % len(traj_ids)
        selected_traj_ids = []
        
        # Try to select different trajectories
        for i in range(model.num_agents):
            idx = (start_idx + i) % len(traj_ids)
            selected_traj_ids.append(traj_ids[idx])
        
        # Create separate save directory for each test
        test_save_dir = os.path.join(args.save_dir, f"test_{test_idx+1}")
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
        
        # Test multi-agent
        response, actions, instructions, used_traj_ids, actual_actions = test_multi_agent_trajectories(
            model, dataset, selected_traj_ids, test_save_dir
        )
        
        # Save results
        with open(os.path.join(test_save_dir, "results.txt"), "w") as f:
            f.write(f"Test {test_idx+1}:\n")
            f.write("Used trajectory IDs:\n")
            for i, traj_id in enumerate(used_traj_ids):
                f.write(f"Agent {i+1}: {traj_id}\n")
            
            f.write("\nInstructions:\n")
            for i, instr in enumerate(instructions):
                f.write(f"Agent {i+1}: {instr}\n")
            
            f.write(f"\nModel response:\n{response}\n\n")
            
            f.write("Parsed actions:\n")
            for agent_idx, action in actions.items():
                action_name = config.model.action_map.get(action, f"Unknown action {action}")
                f.write(f"Agent {agent_idx+1}: {action} ({action_name})\n")
            
            f.write("\nActual actions:\n")
            for agent_idx, action in actual_actions.items():
                action_name = config.model.action_map.get(action, f"Unknown action {action}")
                f.write(f"Agent {agent_idx+1}: {action} ({action_name})\n")
    
    logger.info(f"Test completed. Results saved in {args.save_dir}")

if __name__ == "__main__":
    main() 