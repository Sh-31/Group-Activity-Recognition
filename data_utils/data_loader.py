# 
#  Data Loader Function:

# Group Activity Data Loader:
# 	1. Can return a full image of target frame with its group label (frame, tensor(8)) *needed for B1*. 
# 	2. Can return a all player crops of the target frame with its group label all player have same label  ( (12, crop frame), tensor(1,8)) *needed for B3 step B, C*.
# 	3. Can return a full clip with each frame dir with its group label (all the same) ((9, frame) , tensor(9,8)) *needed for B4*.
#   4. Can return a full clip with all player crop with its group label (all the same) ((12, 9, crop frame), tensor(9,8)) *needed for B5, B6, B7, B8*.

#  Person Activity Data Loader:
# 	1. Can return crop of player image frames in independent way (crop frame , tensor(9)) *needed for B3 step A , B6*.
# 	2. Can return crop of player in the same clip (12 , 9, crop frame) , (tensor(12, 9, 9)) *needed for B5, B7*.
	
# Note: frame means all image dim (C, H, W).
# 

import os
import cv2
import pickle
import torch
from typing import List
from boxinfo import BoxInfo
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

dataset_root = "/teamspace/studios/this_studio/Group-Activity-Recognition/data"
annot_path =   f"{dataset_root}/annot_all.pkl"
videos_path =  f"{dataset_root}/videos"

people_activity_clases = ["Waiting", "Setting", "Digging", "Falling" ,"Spiking"	, "Blocking", "Jumping"	, "Moving", "Standing"]
people_activity_labels  = {class_name.lower():i for i, class_name in enumerate(people_activity_clases)}

group_activity_clases = ["r_set", "r_spike" , "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l_spike", "l_set"]
group_activity_labels  = {class_name:i for i, class_name in enumerate(group_activity_clases)}

# print(people_activity_lables)
# print(group_activity_labels)

class People_Activity(Dataset):
    def __init__ (self, videos_path=videos_path, annot_path=annot_path, seq=False, split=None, F=None, transform=None):
        '''
    
        seq - flag: return all clip dir frames as seqance     

        split - list:
            - Training Set: [1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54].
            - Validation Set: [0 2 8 12 17 19 24 26 27 28 30 33 46 49 51].
            - Testing Set: [4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47].

        transform : for images reprocessing
        '''
        with open(annot_path, 'rb') as file:
            self.videos_annot = pickle.load(file)

        self.data = [] # list of tuple 

        for clip_id in split:

            for clip_dir in self.videos_annot[str(clip_id)].keys():

                dir_frames = list(self.videos_annot[str(clip_id)][str(clip_dir)]['frame_boxes_dct'].items())

                clip_frames , clip_label = [], [] 

                for frame_i, (frame_id , boxes) in enumerate(dir_frames):

                    boxes: List[BoxInfo] = boxes
                    # print(frame_id, boxes)
                    frame = cv2.imread(f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg")

                    boxes_of_the_frames, label_of_the_boxes = [], []

                    for box_i ,frame_box in enumerate(boxes):
                        
                        x_min, y_min, x_max, y_max = frame_box.box
                        cropped_bbox = frame[y_min:y_max, x_min:x_max]

                        # validation of bbox
                        # print(f"clip_id: {str(clip_id)} - dir_id: {str(clip_dir)}  - frame number: {frame_i} - box_number {box_i}")
                        # print(frame_box.category)
                        # image_rgb = cv2.cvtColor(cropped_bbox, cv2.COLOR_BGR2RGB)
                        # plt.imshow(image_rgb)
                        # plt.axis('off')  # Hide axes for better view
                        # plt.show()    
                
                        if transform:
                            cropped_bbox = transform(cropped_bbox)

                        if seq: # if true we will gather all 12 boxes of the frame
                             label_y = torch.zeros((len(labels))) # one hot encoding
                             label_y[labels[frame_box.category]] = 1  
                             boxes_of_the_frames.append(cropped_bbox)
                             label_of_the_boxes.append(label_y) 
                           
                        else: # if false each box is independent
                             label_y = torch.zeros((len(labels))) # one hot encoding
                             label_y[labels[frame_box.category]] = 1  
                             self.data.append((cropped_bbox, label_y ))  # add box, player activity label     

                    if seq: # if true we gather all 9 frames here 
                            stacked_boxes = torch.stack(boxes_of_the_frames)
                            stacked_labels = torch.stack(label_of_the_boxes)
                            clip_frames.append(stacked_boxes)
                            clip_label.append(stacked_labels)

                if seq: # if true we add tuple of (12, 9, 3, 224, 224) , (12, 9, 9) 
                        # Rearrange dimensions to (12, 9, C, H, W) for clip_frames_tensor
                        clip_frames_tensor = torch.stack(clip_frames).permute(1, 0, 2, 3, 4)
                        # Rearrange dimensions to (12, 9, num_labels) for clip_label_tensor
                        clip_label_tensor = torch.stack(clip_label).permute(1, 0, 2)
                        self.data.append((clip_frames_tensor, clip_label_tensor))  
                # print(self.data[0][0].shape)  # torch.Size([12, 9, 3, 224, 224])
                # print(self.data[0][1].shape)  # torch.Size([12, 9, 9])           
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[0]
   



# class Group_Activity(Dataset):
#     def __init__ (self, videos_path=videos_path, annot_path=annot_path, seq=False, crops=False, split=None ,transform=None):
#         '''
    
#         seq - flag: return all clip dir frames as seqance     

#         crop - flag : return the frame or the crops od bbox as seqance


#         split - list:#             - Training Set: [1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54].
#             - Validation Set: [0 2 8 12 17 19 24 26 27 28 30 33 46 49 51].
#             - Testing Set: [4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47].

#         transform : for images reprocessing

#         note can not be seq and crop flag true 

#         '''


#         with open(annot_path, 'rb') as file:
#             self.videos_annot = pickle.load(file)

#         self.transform = transform

#         self.data = []

#         for clip_id in split:

#             for clip_dir in self.videos_annot[str(clip_id)].keys():
                
#                 for clip_dir_frames_ids in  self.videos_annot[str(clip_id)][str(clip_dir)].keys():

#                     image = cv2.imread(f"{videos_path}/{clip_id}/{clip_dir}/clip_dir_frames_ids")
#                     cv2.imshow(image)
               