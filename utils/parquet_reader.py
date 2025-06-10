import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import tqdm
import os

import io
import pandas as pd

import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)
datasets.utils.logging.set_verbosity_info()

class ParquetReader:
    def __init__(self, config):
        '''
        file path like: ./data/train/env_xxx/astar_data/high_long/xxx.parquet
        '''
        self.file_root = config.data.file_root
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if getattr(config.data, 'normalize_images', True) else None
        self.batch_size = getattr(config.data, 'batch_size', 32)
        self.num_workers = getattr(config.data, 'num_workers', 4)
        self.shuffle = getattr(config.data, 'shuffle', True)
        self.env_id = getattr(config.data, 'env_id', 'env_airsim_16')

    def read_parquet(self, split='train'):
        """
        Read parquet files and create DataLoader for certain split
        
        Args:
            split: dataset split ('train', 'val', 'test')
            
        Returns:
            DataLoader for the dataset
        """
        logger.info(f"Creating {split} dataloader...")
        
        # load huggingface dataset to get instruction
        openfly_dataset = load_dataset("IPEC-COMMUNITY/OpenFly", split='train', num_proc=os.cpu_count()//2)

        gpt_instructions = self._extract_instruction(openfly_dataset)

        # Get directories based on split
        if hasattr(self.config.data, f"{split}_path"):
            root_dirs = getattr(self.config.data, f"{split}_path")
        else:
            root_dirs = [os.path.join(self.file_root, dir_name) 
                        for dir_name in os.listdir(self.file_root) 
                        if os.path.isdir(os.path.join(self.file_root, dir_name))]
        
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        # Create dataset
        dataset = TrajectoryDataset(
            root_dirs=root_dirs,
            transform=self.transform,
            instructions=gpt_instructions
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle if split == 'train' else False,
            drop_last=split == 'train',
            pin_memory=True
        )
        
        logger.info(f"Created {split} dataloader with {len(dataset)} samples")
        return dataloader

    def _extract_instruction(self, dataset):
        '''
        Extract instruction from huggingface dataset
        '''
        instructions = {}
        matched = 0

        for item in tqdm.tqdm(dataset, desc="Extracting instructions"):
            if 'image_path' in item and self.env_id in item['image_path']:
                matched += 1
                path_parts = item['image_path'].split('/') # 'image_path' like 'env_airsim_16/astar_data/high_average/2025-1-9_16-8-22_1804289383'
                if len(path_parts) > 0:
                    traj_id = path_parts[-1].split('.')[0] # '2025-1-9_16-8-22_1804289383'
                    if 'gpt_instruction' in item and item['gpt_instruction']:
                        instructions[traj_id] = item['gpt_instruction'] # '2025-1-9_16-8-22_1804289383': 'go to the target'
        
        logger.info(f"Matched {matched} items")
        return instructions

class TrajectoryDataset(Dataset):
    '''
    Initialize trajectory dataset
    :param root_dirs: list of root directories
    :param transform: data augmentation
    :param instructions: dict mapping traj_id to instruction
    '''
    def __init__(self, root_dirs, transform=None, instructions=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.file_paths = []
        self.metadata_df = None
        self.frames_df = None
        self.instructions = instructions or {}
        self._preload_metadata()
    
    def _preload_metadata(self):
        '''
        Preload metadata from all parquet files
        '''
        self.file_paths = []

        root_dirs = self.root_dirs
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        for root_dir in root_dirs:
            if os.path.isdir(root_dir):
                for fname in os.listdir(root_dir):
                    if fname.endswith(".parquet"):
                        full_path = os.path.join(root_dir, fname)
                        self.file_paths.append(full_path)
            else:
                logger.warning(f"dir not found: {root_dir}")

        logger.info(f"Found {len(self.file_paths)} parquet files")
        
        if not self.file_paths:
            logger.error("No parquet files found")
            self.metadata_df = pd.DataFrame(columns=['traj_id', 'file_path', 'num_frames', 'instruction'])
            self.frames_df = pd.DataFrame(columns=['traj_id', 'file_path', 'frame_index'])
            return

        data = []
        count_with_ins = 0
        for i, file_path in enumerate(tqdm.tqdm(self.file_paths, desc="Preloading metadata")):
            try:

                reader = pq.ParquetFile(file_path)
                num_frames = reader.metadata.num_rows
                
                table = pq.read_table(file_path, columns=['traj_id'])
                traj_id = table['traj_id'][0].as_py()

                if traj_id in self.instructions:
                    instruction = self.instructions[traj_id]
                    count_with_ins += 1
                else:
                    instruction = ''
                
                data.append({
                    'traj_id': traj_id,
                    'file_path': file_path,
                    'num_frames': num_frames,
                    'instruction': instruction
                })
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

        logger.info(f"count_with_ins: {count_with_ins}")
        self.metadata_df = pd.DataFrame(data)
        
        if len(self.metadata_df) > 0:
            frames_data = []
            for _, row in self.metadata_df.iterrows():
                file_frames = pd.DataFrame({
                    'traj_id': [row['traj_id']] * row['num_frames'],
                    'file_path': [row['file_path']] * row['num_frames'],
                    'frame_index': range(row['num_frames'])
                })
                frames_data.append(file_frames)
            
            self.frames_df = pd.concat(frames_data, ignore_index=True)
            logger.info(f"Loaded {len(self.frames_df)} frames from {len(self.metadata_df)} trajectories")
        else:
            logger.warning("No valid metadata found")
            self.frames_df = pd.DataFrame(columns=['traj_id', 'file_path', 'frame_index'])
    
    def __len__(self):
        return len(self.frames_df) if hasattr(self, 'frames_df') and self.frames_df is not None else 0
    
    def __getitem__(self, idx):
        '''
        Get single trajectory data point
        '''
        try:
            record = self.frames_df.iloc[idx]
            file_path = record['file_path']
            frame_index = record['frame_index']
            traj_id = record['traj_id']

            table = pq.read_table(
                file_path,
                filters=[('frame_index', '==', frame_index)],
                columns=['image', 'pos', 'yaw', 'action_type', 'action_value']
            )
            row_data = table.to_pandas().iloc[0]
            
            # Process image data
            image_bytes = row_data['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes))

            if self.transform:
                image = self.transform(image)

            traj_metadata = self.metadata_df[self.metadata_df['traj_id'] == traj_id]
            if not traj_metadata.empty:
                traj_instruction = traj_metadata['instruction'].iloc[0]
            else:
                traj_instruction = ""

            # Create sample dictionary
            sample = {
                'image': image,
                'instruction': traj_instruction,
                'position': torch.tensor(row_data['pos'], dtype=torch.float),
                'yaw': torch.tensor(row_data['yaw'], dtype=torch.float),
                'action_type': row_data['action_type'],
                'action_value': torch.tensor(row_data['action_value'], dtype=torch.long),
                'traj_id': traj_id,
                'frame_index': frame_index
            }

            return sample

        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            # Return a default sample in case of error
            return {
                'image': torch.zeros((3, 224, 224)),
                'instruction': '',
                'position': torch.zeros(3),
                'yaw': torch.tensor(0.0),
                'action_type': 'none',
                'action_value': torch.tensor(0),
                'traj_id': '',
                'frame_index': -1
            }
            
    def get_trajectory(self, traj_id):
        '''
        Get all frames from a specific trajectory
        
        Args:
            traj_id: trajectory ID
            
        Returns:
            List of samples from the trajectory in order
        '''
        traj_indices = self.frames_df[self.frames_df['traj_id'] == traj_id].sort_values('frame_index').index.tolist()
        return [self[idx] for idx in traj_indices]