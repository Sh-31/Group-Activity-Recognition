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
        self.feature_extraction = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.lstm = nn.LSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                        )

        self.fc = nn.Sequential(
            nn.Linear(input_size + hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch, 9, 3, 244, 244)
        b, seq, c, h, w = x.shape
        x1 = x.view(b * seq, c, h, w)  # (batch * 9, 3, 244, 244)

        x1 = self.feature_extraction(x1)  # (batch * 9, 2048, 1, 1)
        x1 = x1.view(b, seq, -1)  # (batch, 9, 2048)
        x2, (h, c) = self.lstm(x1)  # x: (batch, 9 , hidden_size)
       
        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation of the frame and Lstm temporal representation 
                                      
        x = x[:, -1, :]  # (batch, hidden_size + 2048)
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = None 
        
        if alpha is not None:
            self.alpha = alpha.clone().detach() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha).clone().detach()
            self.alpha.requires_grad_(True)

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets.argmax(1))
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            focal_loss = alpha[targets.argmax(1)] * focal_loss
            
        return focal_loss.mean()

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
    MODEL_CONFIG = f"{ROOT}/modeling/configs/Baseline B4.yml"    
    CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 4/outputs/Baseline_B4_V1_20250119_054214/checkpoint_epoch_20.pkl"
   
    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    
    # model_summary(args) # Show model details 
    eval(args, CHECKPOINT_PATH) # eval model against  testset

    # ==================================================
    # Group Activity Baseline 4 eval on testset
    # ==================================================
    # Accuracy : 76.59%
    # Average Loss: 0.8396
    # F1 Score (Weighted): 0.7667

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.69      0.67      0.68       192
    #      r_spike       0.86      0.80      0.83       173
    #       r-pass       0.67      0.73      0.70       210
    #   r_winpoint       0.90      0.83      0.86        87
    #   l_winpoint       0.83      0.91      0.87       102
    #       l-pass       0.73      0.72      0.72       226
    #      l-spike       0.89      0.85      0.87       179
    #        l_set       0.71      0.74      0.73       168

    #     accuracy                           0.77      1337
    #    macro avg       0.78      0.78      0.78      1337
    # weighted avg       0.77      0.77      0.77      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 4/outputs/Group_Activity_Baseline_4_eval_on_testset_confusion_matrix.png