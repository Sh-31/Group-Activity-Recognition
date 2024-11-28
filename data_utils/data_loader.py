"""
Data Loader Function:

Group Activity Data Loader:
	1. Can return a full image of target frame with its group label (frame, tensor(8)) *needed for B1*. 
	2. Can return a all player crops of the target frame with its group label all player have same label  ( (12, crop frame), tensor(1,8)) *needed for B3 step B, C*.
	3. Can return a full clip with each frame dir with its group label (all the same) ((9, frame) , tensor(9,8)) *needed for B4*.
    4. Can return a full clip with all player crop with its group label (all the same) ((12, 9, crop frame), tensor(9,8)) *needed for B5, B6, B7, B8*.

Person Activity Data Loader:
	1. Can return crop of player image frames in independent way (crop frame , tensor(9)) *needed for B3 step A , B6*.
	2. Can return crop of player in the same clip (12 , 9, crop frame) , (tensor(12, 9, 9)) *needed for B5, B7*.
	
 Note:
    1.  frame and crop frame means all image dim (C, H, W).
    2.  Group Activity Data Loader Case 1 and 2 only utilize target frame of each clip

"""
import cv2
import pickle
import torch
import numpy as np
import albumentations as A
from typing import List, Tuple
from data_utils.boxinfo import BoxInfo
from torch.utils.data import Dataset
from pathlib import Path

dataset_root = "/teamspace/studios/this_studio/Group-Activity-Recognition/data"
annot_path = f"{dataset_root}/annot_all.pkl"
videos_path = f"{dataset_root}/videos"

person_activity_clases = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_clases)}

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}


class Person_Activity_DataSet(Dataset):
    def __init__(self, videos_path: str, annot_path: str, seq: bool = False, 
                 split: list = [], labels: dict = {}, only_tar: bool = False, 
                 transform=None):
        """
        Args:
            videos_path: Path to video frames
            annot_path: Path to annotations file
            seq: Whether to return sequential data
            split: List of clip IDs to use
            labels: Person activity labels dictionary
            only_tar: Whether to use only target frames
            transform:  transform to apply
        """
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.seq = seq
        self.only_tar = only_tar
        self.labels = labels
       
        
        # Load annotations but store only metadata
        with open(annot_path, 'rb') as f:
            videos_annot = pickle.load(f)
            
        # Create index mapping for efficient retrieval
        self.frame_indices = []
        for clip_id in split:
            clip_data = videos_annot[str(clip_id)]
            for clip_dir in clip_data.keys():
                frames_data = clip_data[str(clip_dir)]['frame_boxes_dct']
                
                if only_tar and str(frame_id) != str(clip_dir): continue
                            
                if seq:
                    # Group frames for sequence processing
                    sequence_frames = []
                    for frame_id, boxes in frames_data.items():
                        
                        sequence_frames.append({
                            'frame_id': frame_id,
                            'boxes': boxes
                        })
                    
                    if sequence_frames:
                        self.frame_indices.append({
                            'clip_id': clip_id,
                            'clip_dir': clip_dir,
                            'frames': sequence_frames,
                            'type': 'sequence'
                        })
                else:
                    # Store individual frame metadata
                    for frame_id, boxes in frames_data.items():
                        
                        for box in boxes:
                            self.frame_indices.append({
                                'clip_id': clip_id,
                                'clip_dir': clip_dir,
                                'frame_id': frame_id,
                                'box': box,
                                'type': 'single'
                            })

    def __len__(self):
        return len(self.frame_indices)
    
    def load_and_transform_person(self, frame: np.ndarray, box: BoxInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and transform person bbox from frame"""
        x_min, y_min, x_max, y_max = box.box
        person_crop = frame[y_min:y_max, x_min:x_max]
        
       
        if self.transform:
            transformed = self.transform(image=person_crop)
            person_crop = transformed['image']
            
        # Create one-hot encoded label
        label = np.zeros(len(self.labels))
        label[self.labels[box.category]] = 1
        
        return person_crop, label

    def load_frame(self, clip_id: str, clip_dir: str, frame_id: str) -> np.ndarray:
        """Load a single frame"""
        frame_path = self.videos_path / str(clip_id) / str(clip_dir) / f"{frame_id}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        return frame

    def __getitem__(self, idx):
        sample = self.frame_indices[idx]
        
        if sample['type'] == 'sequence':
            # Process sequence of frames
            sequence_crops = []
            sequence_labels = []
            
            for frame_data in sample['frames']:
                frame = self.load_frame(
                    sample['clip_id'], 
                    sample['clip_dir'], 
                    frame_data['frame_id']
                )
                
                frame_crops = []
                frame_labels = []
                
                for box in frame_data['boxes']:
                    person_crop, label = self.load_and_transform_person(frame, box)
                    frame_crops.append(person_crop)
                    frame_labels.append(label)
                
                if frame_crops:
                    sequence_crops.append(np.stack(frame_crops))
                    sequence_labels.append(np.stack(frame_labels))
            
            if sequence_crops:
                # Stack and transpose to get (num_people, num_frames, C, H, W)
                crops_tensor = np.stack(sequence_crops)
                crops_tensor = np.transpose(crops_tensor, (1, 0, 2, 3, 4))
                
                # Stack and transpose labels to get (num_people, num_frames, num_classes)
                labels_tensor = np.stack(sequence_labels)
                labels_tensor = np.transpose(labels_tensor, (1, 0, 2))
                
                return torch.from_numpy(crops_tensor), torch.from_numpy(labels_tensor)
            
        else:
            # Process single frame
            frame = self.load_frame(
                sample['clip_id'], 
                sample['clip_dir'], 
                sample['frame_id']
            )
            
            person_crop, label = self.load_and_transform_person(frame, sample['box'])

            return  (person_crop,  torch.from_numpy(label))
                   
             
class Group_Activity_DataSet(Dataset):
    def __init__(self, videos_path: str, annot_path: str, seq: bool = False, 
                 crops: bool = False, split: list = [], only_tar: bool= False,
                 labels: dict = {}, transform=None):
        """
        Args:
            videos_path: Path to video frames
            annot_path: Path to annotations file
            seq: Whether to return sequential data
            crops: Whether to return person crops or full frames
            split: List of clip IDs to use
            labels: Group activity labels dictionary
            transform: Optional transform to apply
            only_tar: Whether to use only target frames
        """
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.seq = seq
        self.crops = crops
        self.labels = labels
        self.only_tar = False
        
        # Load annotations and store only metadata
        with open(annot_path, 'rb') as f:
            videos_annot = pickle.load(f)
            
        # Create index mapping for efficient retrieval
        self.data = []
        for clip_id in split:
            clip_dirs = videos_annot[str(clip_id)]
            
            for clip_dir in clip_dirs.keys():
                frames_data = clip_dirs[str(clip_dir)]['frame_boxes_dct']
                category = clip_dirs[str(clip_dir)]['category']
                dir_frames = list(clip_dirs[str(clip_dir)]['frame_boxes_dct'].items())
              
                if not crops and not seq:
                    # return a full image of target frame with its group label (frame, tensor(8))
                    for frame_id , boxes in dir_frames:

                        if only_tar and str(frame_id) != str(clip_dir): continue
                     
                        self.data.append({
                            'frame_path': f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg",
                            'category': category,
                        })

                elif not crops and seq:
                     # return a full clip with each frame dir with its group label (all the same) ((9, frame) , tensor(9,8))
                     frames_paths = []
                     for frame_id , boxes in dir_frames:

                          if only_tar and str(frame_id) != str(clip_dir): continue
                          
                          frames_paths.append(f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg")
                   
                     self.data.append({
                        'frames_paths':frames_paths,
                        'category': category,
                        })

                elif crops and not seq:
                    # return a all player crops of the target frame with its group label (all player have same label)  ( (12, crop frame), tensor(1,8))                   
                    for frame_id , boxes in dir_frames:

                        if only_tar and str(frame_id) != str(clip_dir): continue

                        frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
                        boxes: List[BoxInfo] = boxes
                        frames_boxes = []

                        for box in boxes:
                            frames_boxes.append(box)
                            
                        self.data.append({
                        'frame_path': frame_path,
                        'boxes': boxes,
                        'category': category,
                        })    
                       
                       
                else:  
                      # when crop and seq are true return a full clip with all player crop with its group label (all the same) ((12, 9, crop frame), tensor(9,8)
                      frame_data = []

                      for frame_id , boxes in dir_frames:

                         if only_tar and str(frame_id) != str(clip_dir): continue

                         frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"

                         frames_boxes = []
                         for box in boxes:
                            frames_boxes.append(box)

                         frame_data.append((frame_path, frames_boxes))
                         self.data.append({
                            'frame_data':frame_data,
                            'category': category,
                        })      


    def __len__(self):
        return len(self.data)
    

    def extract_person_crops(self, frame: np.ndarray, boxes: List[BoxInfo]):
        """Extract and transform person crops from frame"""
        crops: List = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.box
          
            person_crop = frame[y_min:y_max, x_min:x_max]
        
            if self.transform:
                transformed = self.transform(image=person_crop)
                person_crop = transformed['image']
                
            crops.append(person_crop)
    
        return torch.stack(crops)

    def __getitem__(self, idx):
        sample = self.data[idx]
        num_class = len(self.labels)
        label = torch.zeros(num_class) 
        label[self.labels[sample['category']]] = 1

        if not self.crops and not self.seq:
            frame = cv2.imread(sample['frame_path'])
          
            if self.transform:
                frame = self.transform(image=frame)['image']

            return  frame, label

        elif not self.crops and self.seq:
             # return a full clip with each frame dir with its group label (all the same) ((9, frame) , tensor(9,8))
            clip = [] 
            labels = []
            for frame_path in sample['frames_paths']:
                frame = cv2.imread(frame_path)
                labels.append(label)
                
                if self.transform:
                    frame = self.transform(image=frame)['image']
                clip.append(frame)

            return torch.stack(clip) , torch.stack(labels)
                 
        elif self.crops and not self.seq:
             # return a all player crops of the target frame with its group label (all player have same label)  ( (12, crop frame), tensor(1,8)) 
             frame = cv2.imread(sample['frame_path'])
             crops = self.extract_person_crops(frame, sample['boxes'])  
             return crops, label  

        else:
             # when crop and seq are true return a full clip with all player crop with its group label (all the same) ((12, 9, crop frame), tensor(9,8)
             clip = []
             labels = []

             for frame_path, boxes in sample['frame_data']:
                frame = cv2.imread(frame_path)
                crops = self.extract_person_crops(frame, boxes) 
                clip.append(crops)
                labels.append(label) 

             # Rearrange dimensions to (12, 9, C, H, W) for clip_frames_tensor  
             clip = torch.stack(clip).permute(1, 0, 2, 3, 4) 
             labels = torch.stack(labels)

             return clip , labels
