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

logger = logging.getLogger(__name__)

class ParquetReader:
    def __init__(self, config):
        '''
        file path like: ./data/train/env_xxx/high_long/xxx.parquet
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

    def read_parquet(self, split='train'):
        """
        Read parquet files and create DataLoader
        
        Args:
            split: dataset split ('train', 'val', 'test')
            
        Returns:
            DataLoader for the dataset
        """
        logger.info(f"Creating {split} dataloader...")
        
        # Get directories based on split
        if hasattr(self.config.data, f"{split}_path"):
            root_dirs = getattr(self.config.data, f"{split}_path")
        else:
            root_dirs = [os.path.join(self.file_root, dir_name) 
                        for dir_name in os.listdir(self.file_root) 
                        if os.path.isdir(os.path.join(self.file_root, dir_name))]
            
        # Create dataset
        dataset = TrajectoryDataset(
            root_dirs=root_dirs,
            transform=self.transform
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

class TrajectoryDataset(Dataset):
    '''
        Initialize trajectory dataset
        :param root_dirs: list of root directories
        :param transform: data augmentation
    '''
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.file_paths = []
        self.metadata_df = None
        self._preload_metadata()
    
    def _preload_metadata(self):
        '''
        Preload metadata from all parquet files
        '''
        self.data_records = []

        for root_dir in self.root_dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if fname.endswith(".parquet"):
                        full_path = os.path.join(dirpath, fname)
                        self.file_paths.append(full_path)

        logger.info(f"Found {len(self.file_paths)} parquet files")

        # Process each file and collect metadata
        all_metadata = []
        for file_path in tqdm.tqdm(self.file_paths, desc="Preloading metadata"):
            try:
                table = pq.read_table(file_path)
                metadata = table.select(['traj_id', 'image_id', 'frame_index']).to_pandas()
                metadata['file_path'] = file_path
                metadata['row_index'] = metadata.index.values
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

        # Combine all metadata into a single DataFrame
        if all_metadata:
            self.metadata_df = pd.concat(all_metadata, ignore_index=True)
            logger.info(f"Loaded metadata for {len(self.metadata_df)} frames from {len(self.file_paths)} trajectories")
        else:
            logger.warning("No valid metadata found")
            self.metadata_df = pd.DataFrame(columns=['traj_id', 'image_id', 'frame_index', 'file_path', 'row_index'])

    def __len__(self):
        return len(self.metadata_df) if self.metadata_df is not None else 0
    
    def __getitem__(self, idx):
        '''
        Get single trajectory data point
        '''
        try:
            record = self.metadata_df.iloc[idx]
            file_path = record['file_path']
            # row_index = record['row_index']

            # Read specific row from parquet file
            table = pq.read_table(
                file_path,
                filters=[('frame_index', '==', record['frame_index'])],
                columns=['image', 'instruction', 'pos', 'yaw', 'action_type', 'action_value']
            )
            row_data = table.to_pandas().iloc[0]
            
            # Process image data
            image_bytes = row_data['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes))

            if self.transform:
                image = self.transform(image)

            # Create sample dictionary
            sample = {
                'image': image,
                'instruction': row_data['instruction'],
                'position': torch.tensor(row_data['pos'], dtype=torch.float),
                'yaw': torch.tensor(row_data['yaw'], dtype=torch.float),
                'action_type': row_data['action_type'],
                'action_value': torch.tensor(row_data['action_value'], dtype=torch.long),
                'traj_id': record['traj_id'],
                'frame_index': record['frame_index']
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
        traj_indices = self.metadata_df[self.metadata_df['traj_id'] == traj_id].sort_values('frame_index').index.tolist()
        return [self[idx] for idx in traj_indices]