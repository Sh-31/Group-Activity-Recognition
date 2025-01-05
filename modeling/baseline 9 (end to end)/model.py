"""
Baseline 9 Description:
In previous  baselines, person activity and group activity losses were handled independently, resulting in a two-stage model. 
Baseline 9 combines these processes into a unified, end-to-end training pipeline. This approach allows for the simultaneous 
optimization of both person-level and group-level activity classification using a shared gradient flow.
"""
import os
import sys
import torch
import argparse
import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary

class Hierarchical_Group_Activity_Classifer(nn.Module):
    def __init__(self, person_num_classes, group_num_classes, hidden_size, num_layers):
        super(Hierarchical_Group_Activity_Classifer, self).__init__()

        self.feature_extractor = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1])
        
        self.layer_norm_1 = nn.LayerNorm(2048)
        
        self.lstm_1 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, person_num_classes)
        )

        self.layer_norm_2 = nn.LayerNorm(2048)
        self.pool = nn.AdaptiveMaxPool2d((1, 1024))
     
        self.lstm_2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, group_num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape  # seq => frames
        x = x.view(b*bb*seq, c, h, w)  # (b * bb * seq, c, h, w)
        x1 = self.feature_extractor(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1)       # (batch * bbox, seq, 2048)
        x1 = self.layer_norm_1(x1)          # (batch * bbox, seq, 2048)
        x2, (h_1 , c_1) = self.lstm_1(x1) # (batch * bbox, seq, hidden_size)

        y1 = self.fc_1(x2[:, -1, :])  # (batch, person_num_classes)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene (same as paper).
       
        x = x.view(b*seq, bb, -1) # (batch * seq, bbox, hidden_size)
        team_1 = x[:, :6, :]      # (batch * seq, 6, hidden_size)
        team_2 = x[:, 6:, :]      # (batch * seq, 6, hidden_size)

        team_1 = self.pool(team_1) # (batch * seq, 1, 1024)
        team_2 = self.pool(team_2) # (batch * seq, 1, 1024)
        x = torch.cat([team_1, team_2], dim=1)  # (batch * seq, 2, 1024)
       
        x = x.view(b, seq, -1) # (batch, seq, 2048)
        x = self.layer_norm_2(x) # (batch, seq, 2048)
        x, (h_2 , c_2) = self.lstm_2(x) # (batch, seq, hidden_size)

        x = x[:, -1, :]     # (batch, hidden_size)
        y2 = self.fc_2(x)   # (batch, group_num_classes)
        return {'person_output': y1, 'group_output': y2}


def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, person_labels, group_labels  = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_person_labels = []

    for clip, label in zip(clips, person_labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_person_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_person_labels = torch.stack(padded_person_labels)
    group_labels = torch.stack(group_labels)
    
    group_labels = group_labels[:,-1, :] # # utils the label of last frame
    padded_person_labels = padded_person_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_person_labels.shape # batch, bbox, num_clases
    padded_person_labels = padded_person_labels.view(b*bb, num_class)

    return padded_clips, padded_person_labels, group_labels

if __name__ == "__main__":
    pass
