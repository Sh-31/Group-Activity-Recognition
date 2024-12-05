"""
Baseline 4 Description :
--------------------------------
Temporal model with image features level 
be  representation per clip use 9 frames 
per image then we have sequence for each clip of 9 steps 
train an LSTM on these sequences.
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

class Group_Activity_Temporal_Classifer(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(Group_Activity_Temporal_Classifer, self).__init__()
        
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.feature_extraction = nn.Sequential(
            *list(resnet50.children())[:-1], # remove fc layer
            nn.Dropout(0.5)  
        )
        
        self.lstm = nn.LSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                        )

        self.fc =  nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch, 9, 3, 244, 244)
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)  # (batch * 9, 3, 244, 244)

        x = self.feature_extraction(x)  # (batch * 9, 2048, 1, 1)
        x = x.view(b, seq, -1)  # (batch, 9, 2048)
        
        x, (h, c) = self.lstm(x)  # x: (batch, 9 , hidden_size)
        x = x[:, -1, :]  # (64, hidden_size)
        x = self.fc(x)  # (64, num_classes)
        
        return x

def collate_fn(batch):
    clips, labels = zip(*batch) 
    clips = torch.stack(clips, dim=0) 
    labels = torch.stack(labels, dim=0)  
    labels = labels[:, -1, :]  # utile the label of last frame
    return clips, labels

def model_summary(args):

    sys.path.append(os.path.abspath(args.project_root))
    from helper_utils import load_config

    config = load_config(args.config_path)

    model = Group_Activity_Temporal_Classifer(
        num_classes=config.model["num_classes"], 
        input_size=config.model["input_size"], 
        hidden_size=config.model["hidden_size"], 
        num_layers=config.model["num_layers"]
    )
    
    summary(model)

def eval(args, checkpoint_path):

    sys.path.append(os.path.abspath(args.project_root))
    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels
    
    config = load_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Group_Activity_Temporal_Classifer(
            num_classes=config.model['num_classes'],
            input_size=config.model['input_size'],
            hidden_size=config.model['hidden_size'],
            num_layers=config.model['num_layers']
        )

    model = load_checkpoint(
                model=model, 
                checkpoint_path=checkpoint_path, 
                device=device, 
                optimizer=None
            )

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
        videos_path=f"{args.project_root}/{config.data['videos_path']}",
        annot_path=f"{args.project_root}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels, 
        transform=test_transforms,
        crops=False,
        seq=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=80,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.project_root}/modeling/baseline 4/outputs"
    prefix = "Group Activity Baseline 4 eval on testset"

    metrics = model_eval(
            model=model, 
            data_loader=test_loader, 
            criterion=criterion, 
            device=device, 
            path=path, 
            prefix=prefix, 
            class_names=config.model["num_clases_label"]
    )

    return metrics

if __name__ == "__main__":
    
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/configs/Baseline B4.yml"    
    CHECKPOINT_PATH = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 4/outputs/Baseline_B4_V1_20241204_213623/checkpoint_epoch_13.pkl"
   
    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    
    # model_summary(args) # Show model details 
    eval(args, CHECKPOINT_PATH) # eval model against  testset

    # ==================================================
    #  Group Activity Baseline 4 eval on testset
    # ==================================================
    # Accuracy : 73.45%
    # Average Loss: 0.8009
    # F1 Score (Weighted): 0.7327

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.68      0.64      0.66       192
    #      r_spike       0.79      0.80      0.80       173
    #       r-pass       0.70      0.67      0.68       210
    #   r_winpoint       0.76      0.74      0.75        87
    #   l_winpoint       0.83      0.88      0.85       102
    #       l-pass       0.70      0.71      0.70       226
    #      l-spike       0.79      0.88      0.83       179
    #        l_set       0.69      0.65      0.67       168

    #     accuracy                           0.73      1337
    #    macro avg       0.74      0.75      0.74      1337
    # weighted avg       0.73      0.73      0.73      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 4/outputs/Group_Activity_Baseline_4_eval_on_testset_confusion_matrix.png