import cv2
import pickle
import torch
import numpy as np
import albumentations as A
from typing import List
from .boxinfo import BoxInfo
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

dataset_root = "/teamspace/studios/this_studio/Group-Activity-Recognition/data"
annot_path = f"{dataset_root}/annot_all.pkl"
videos_path = f"{dataset_root}/videos"

person_activity_clases = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_clases)}

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}

def load_image(image_path):
    """Thread-safe image loading function"""
    try:
        return cv2.imread(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def apply_transform(transform, frame):
      if transform:
        if isinstance(transform, A.Compose):
            transformed = transform(image=frame)
            frame = transformed['image']
        else:
            frame = transform(frame)

      if isinstance(frame, torch.Tensor):
            frame = frame.numpy()  

      return  frame 

class ImageLoader:
    def __init__(self, num_threads=4, queue_size=100):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = {}
        
    def preload_images(self, image_paths):
        """Submit image loading tasks to thread pool"""
        self.futures.update( {path: self.executor.submit(load_image, path) for path in image_paths if path not in self.futures})

    def get_image(self, image_path):
        """Get image from cache or load it"""
        if image_path in self.futures:
            future = self.futures[image_path]
            image = future.result()
            del self.futures[image_path]  # Clean up
            return image
        return load_image(image_path)

def get_frame_paths(clip_id, clip_dir, frame_ids, videos_path):
    """Get all image paths for a clip"""
    return [f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg" 
            for frame_id in frame_ids]

def process_clip_people_activity(args):
    clip_id, videos_path, videos_annot, seq, labels, only_tar, transform, num_threads = args
    data = []
    image_loader = ImageLoader(num_threads=num_threads)
    
    for clip_dir in videos_annot[str(clip_id)].keys():
        dir_frames = list(videos_annot[str(clip_id)][str(clip_dir)]['frame_boxes_dct'].items())
        
        # Preload images for this clip
        frame_paths = get_frame_paths(clip_id, clip_dir, [frame_id for frame_id, _ in dir_frames], videos_path)
        image_loader.preload_images(frame_paths)
        
        clip_frames, clip_label = [], []

        for frame_i, (frame_id, boxes) in enumerate(dir_frames):
            frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
            frame = image_loader.get_image(frame_path)
            
            # Use only target frames 
            if only_tar and str(frame_id) != str(clip_dir): continue
                    
            if frame is None:
                continue
                
            boxes: List[BoxInfo] = boxes
            boxes_of_the_frames, label_of_the_boxes = [], []

            for frame_box in boxes:
                x_min, y_min, x_max, y_max = frame_box.box
                cropped_bbox = frame[y_min:y_max, x_min:x_max]

                cropped_bbox = apply_transform(transform, cropped_bbox)
                    
                if seq:
                    label_y = np.zeros(len(labels))
                    label_y[labels[frame_box.category]] = 1
                    boxes_of_the_frames.append(cropped_bbox)
                    label_of_the_boxes.append(label_y)
                else:
                    label_y = np.zeros(len(labels))
                    label_y[labels[frame_box.category]] = 1
                    data.append((cropped_bbox, label_y))

            if seq and boxes_of_the_frames:
                stacked_boxes = np.stack(boxes_of_the_frames)
                stacked_labels = np.stack(label_of_the_boxes)
                clip_frames.append(stacked_boxes)
                clip_label.append(stacked_labels)

        if seq and clip_frames:
            clip_frames_tensor = np.stack(clip_frames)
            # Rearrange dimensions to (12, 9, C, H, W) for clip_frames_tensor
            clip_frames_tensor = np.transpose(clip_frames_tensor, (1, 0, 2, 3, 4))
            clip_label_tensor = np.stack(clip_label)
            # Rearrange dimensions to (12, 9, num_labels) for clip_label_tenso
            clip_label_tensor = np.transpose(clip_label_tensor, (1, 0, 2))
            data.append((clip_frames_tensor, clip_label_tensor))

    return data

def process_clip_group_activity(args):
    clip_id, videos_path, videos_annot, seq, crops, labels, transform, num_threads = args
    data = []
    image_loader = ImageLoader(num_threads=num_threads)
    
    clip_dirs_keys = videos_annot[str(clip_id)].keys()
    clip_dirs = videos_annot[str(clip_id)]

    for clip_dir in clip_dirs_keys:
        dir_frames = list(clip_dirs[str(clip_dir)]['frame_boxes_dct'].items())
        
        # Preload images for this clip
        frame_paths = get_frame_paths(clip_id, clip_dir, [frame_id for frame_id, _ in dir_frames], videos_path)
        image_loader.preload_images(frame_paths)
        
        category = clip_dirs[str(clip_dir)]['category']
        label_y = np.zeros(len(labels))
        label_y[labels[category]] = 1

        if not crops and not seq:
            frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{clip_dir}.jpg"
            frame = image_loader.get_image(frame_path)
            if frame is not None:
                frame = apply_transform(transform, frame)
                data.append((frame, label_y))

        elif not crops and seq:
            dir_frames_pro, dir_frames_labels = [], []
            
            for frame_id, boxes in dir_frames:
                frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
                frame = image_loader.get_image(frame_path)
                if frame is not None:
                    frame = apply_transform(transform, frame)
                    dir_frames_pro.append(frame)
                    dir_frames_labels.append(label_y)

            if dir_frames_pro:
                stacked_frames = np.stack(dir_frames_pro)
                stacked_labels = np.stack(dir_frames_labels)
                data.append((stacked_frames, stacked_labels))

        elif crops and not seq:
            for frame_id, boxes in dir_frames:
                if str(frame_id) != str(clip_dir):
                    continue

                frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
                frame = image_loader.get_image(frame_path)
                if frame is None:
                    continue
                    
                boxes: List[BoxInfo] = boxes
                frames_boxes = []

                for frame_box in boxes:
                    x_min, y_min, x_max, y_max = frame_box.box
                    cropped_bbox = frame[y_min:y_max, x_min:x_max]
                    cropped_bbox = apply_transform(transform, cropped_bbox)
                    frames_boxes.append(cropped_bbox)

                if frames_boxes:
                    stacked_frames_boxes = np.stack(frames_boxes)
                    data.append((stacked_frames_boxes, label_y))

        else:
            clip_frames, clip_label = [], []
            for frame_id, boxes in dir_frames:
                frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
                frame = image_loader.get_image(frame_path)
                
                if frame is None:
                    continue
                    
                boxes: List[BoxInfo] = boxes
                boxes_of_the_frames = []

                for frame_box in boxes:
                    x_min, y_min, x_max, y_max = frame_box.box
                    cropped_bbox = frame[y_min:y_max, x_min:x_max]
                    cropped_bbox = apply_transform(transform, cropped_bbox)
                    boxes_of_the_frames.append(cropped_bbox)

                if boxes_of_the_frames:
                    stacked_boxes = np.stack(boxes_of_the_frames)
                    clip_frames.append(stacked_boxes)
                    clip_label.append(label_y)

            if clip_frames:
                clip_frames_tensor = np.stack(clip_frames)
                clip_frames_tensor = np.transpose(clip_frames_tensor, (1, 0, 2, 3, 4))
                clip_label_tensor = np.stack(clip_label)
                data.append((clip_frames_tensor, clip_label_tensor))

    return data

def numpy_to_torch(data_item):
    if isinstance(data_item, tuple):
        return tuple(torch.from_numpy(item) if isinstance(item, np.ndarray) else item 
                    for item in data_item)
    return data_item

class Person_Activity_DataSet(Dataset):
    def __init__(self, videos_path=videos_path, annot_path=annot_path, seq=False, split:list=[], 
                 labels:dict={}, only_tar=False, transform=None, num_workers=None, num_threads=4):
        with open(annot_path, 'rb') as file:
            self.videos_annot = pickle.load(file)

        if num_workers is None:
            num_workers = min(cpu_count(), len(split))

        process_args = [(clip_id, videos_path, self.videos_annot, seq, labels, only_tar, transform, num_threads) 
                       for clip_id in split]

        with Pool(num_workers) as pool:
            results = pool.map(process_clip_people_activity, process_args)

        self.data = [item for result in results for item in result]
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return numpy_to_torch(self.data[idx])

class Group_Activity_DataSet(Dataset):
    def __init__(self, videos_path=videos_path, annot_path=annot_path, seq=False, crops=False, 
                 split:list=[], labels:dict={}, transform=None, num_workers=None, num_threads=4):
        with open(annot_path, 'rb') as file:
            self.videos_annot = pickle.load(file)

        if num_workers is None:
            num_workers = min(cpu_count(), len(split))

        process_args = [(clip_id, videos_path, self.videos_annot, seq, crops, labels, transform, num_threads) 
                       for clip_id in split]

        with Pool(num_workers) as pool:
            results = pool.map(process_clip_group_activity, process_args)

        self.data = [item for result in results for item in result]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return numpy_to_torch(self.data[idx])