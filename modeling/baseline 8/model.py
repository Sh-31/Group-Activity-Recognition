"""
Baseline 8 Description :
--------------------------------
Full model V2 train LSTM on crops level (LSTM on a player)
extract clips: sequence of 9 steps per player for each frame,
max pool each player team in dependent way concatenate features
from both teams along the feature dimension then fed it to LSTM 2 on the frame level
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

class Person_Activity_Temporal_Classifer(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(Person_Activity_Temporal_Classifer, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.layer_norm = nn.LayerNorm(2048)
        
        self.lstm_1 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (batch * bbox * seq, c, h, w)
        x = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x = x.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x = self.layer_norm(x)
        x, (h , c) = self.lstm_1(x) # (batch * bbox, seq, hidden_size)

        x = x[:, -1, :] # (batch * bbox, hidden_size)
        x = self.fc(x) # (batch * bbox, num_class)  
        
        return x

class Group_Activity_Temporal_Classifer(nn.Module):
    def __init__(self, person_feature_extraction, hidden_size, num_layers, num_classes):
        super(Group_Activity_Temporal_Classifer, self).__init__()

        self.resnet50 = person_feature_extraction.resnet50
        self.lstm_1 = person_feature_extraction.lstm_1

        for module in [self.resnet50, self.lstm_1]:
            for param in module.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d((1, 1024))
        self.layer_norm = nn.LayerNorm(2048)  # Layer normalization for better stability (will be shared through the network)
        
        self.lstm_2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (b * bb *seq, c, h, w)
        x1 = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x1 = self.layer_norm(x1) # (batch * bbox, seq, 2048)
        x2, (h_1 , c_1) = self.lstm_1(x1) # (batch * bbox, seq, hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene (same as paper).
       
        x = x.view(b*seq, bb, -1) # (batch * seq, bbox, hidden_size)
        team_1 = x[:, :6, :]      # (batch * seq, 6, hidden_size)
        team_2 = x[:, 6:, :]      # (batch * seq, 6, hidden_size)

        team_1 = self.pool(team_1) # (batch * seq, 1, 1024)
        team_2 = self.pool(team_2) # (batch * seq, 1, 1024)
        x = torch.cat([team_1, team_2], dim=1)  # (batch * seq, 2, 1024)
       
        x = x.view(b, seq, -1) # (batch, seq, 2048)
        x = self.layer_norm(x) # (batch, seq, 2048)
        x, (h_2 , c_2) = self.lstm_2(x) # (batch, seq, hidden_size)

        x = x[:, -1, :] # (batch, hidden_size)
        x = self.fc(x)  # (batch, num_class)
        return x

def person_collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)
    
    padded_labels = padded_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_labels.shape # batch, bbox, num_clases
    padded_labels = padded_labels.view(b*bb, num_class)

    return padded_clips, padded_labels

def group_collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []

    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            clip = torch.cat((clip, clip_padding), dim=0)
    
        padded_clips.append(clip)
       
    padded_clips = torch.stack(padded_clips)
    labels = torch.stack(labels)
    
    labels = labels[:,-1, :] # utils the label of last frame
    
    return padded_clips, labels

def eval(args, checkpoint_path):

    sys.path.append(os.path.abspath(args.ROOT))
    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels
    
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Group_Activity_Temporal_Classifer(
        person_feature_extraction=person_act_cls, 
        num_classes=config.model['num_classes']['group_activity'],
        hidden_size=config.model['hyper_param']['group_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['group_activity']['num_layers'], 
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
   
    model = model.to(device)

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    test_dataset = Group_Activity_DataSet(
        videos_path=f"{args.ROOT}/{config.data['videos_path']}",
        annot_path=f"{args.ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels, 
        transform=test_transforms,
        crops=True,
        seq=True, 
        sort=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=group_collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.ROOT}/modeling/baseline 8/outputs"
    prefix = "Group Activity Baseline 8 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"]['group_activity'])

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = f"{ROOT}/modeling/configs/Baseline B8.yml"    
    GROUP_ACTIVITY_CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 8/outputs/Baseline_B8_Step_B_V1_2025_01_09_19_29/checkpoint_epoch_63.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    sys.path.append(os.path.abspath(args.ROOT))

    from helper_utils import load_config
    config = load_config(args.config_path)

    person_act_cls = Person_Activity_Temporal_Classifer(
        hidden_size=config.model['hyper_param']['person_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['person_activity']['num_layers'],
        num_classes=config.model['num_classes']['person_activity']
    )

    model = Group_Activity_Temporal_Classifer(
        person_feature_extraction=person_act_cls, 
        num_classes=config.model['num_classes']['group_activity'],
        hidden_size=config.model['hyper_param']['group_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['group_activity']['num_layers'], 
    )

    summary(model)
    eval(args, GROUP_ACTIVITY_CHECKPOINT_PATH)
    # ==================================================
    # Group Activity Baseline 8 eval on testset
    # ==================================================
    # Accuracy : 92.30%
    # Average Loss: 0.3578
    # F1 Score (Weighted): 0.9229

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.95      0.86      0.91       192
    #      r_spike       0.96      0.91      0.93       173
    #       r-pass       0.88      0.95      0.92       210
    #   r_winpoint       0.89      0.90      0.89        87
    #   l_winpoint       0.91      0.92      0.92       102
    #       l-pass       0.93      0.96      0.95       226
    #      l-spike       0.92      0.93      0.92       179
    #        l_set       0.93      0.92      0.93       168

    #     accuracy                           0.92      1337
    #    macro avg       0.92      0.92      0.92      1337
    # weighted avg       0.92      0.92      0.92      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 8/outputs/Group_Activity_Baseline_8_eval_on_testset_confusion_matrix.png