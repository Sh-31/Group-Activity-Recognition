"""
Baseline 5 Description :
--------------------------------
This baseline is a temporal extension of B3, 
where person-specific features pooled over all individuals 
in each frame are fed to an LSTM model 
to capture group dynamics.
"""
import os
import sys
import torch
import pickle 
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
        
        self.lstm = nn.LSTM(
                            input_size=2048,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                        )

        self.fc =  nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (batch * bbox * seq, c, h, w)
        x = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x = x.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x, (h , c) = self.lstm(x) # (batch * bbox, seq, hidden_size)

        x = x[:, -1, :] # (batch * bbox, hidden_size)
        x = self.fc(x) # (batch * bbox, num_class)  
        
        return x

class Group_Activity_Classifer(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(Group_Activity_Classifer, self).__init__()

        self.resnet50 = person_feature_extraction.resnet50
        self.lstm = person_feature_extraction.lstm

        for module in [self.resnet50,  self.lstm]:
            for param in module.parameters():
                param.requires_grad = False
                
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [Batch, 12, hidden_size] -> [Batch, 1, 2048]
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (b * bb * seq, c, h, w)
        x1 = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x2, (h , c) = self.lstm(x1) # (batch * bbox, seq, hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene.
        x = x[:, -1, :]                # (batch * bbox, hidden_size)
        
        x = x.view(b, bb, -1) # (batch , bbox, hidden_size)
        x = self.pool(x) # (batch, 1, 2048)
        x = x.squeeze(dim=1) # (batch, 2048)

        x = self.fc(x) # (batch, num_class)
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

def eval(args, person_activity_checkpoint, checkpoint_path):

    sys.path.append(os.path.abspath(args.ROOT))
    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels
    
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(person_activity_checkpoint, 'rb') as f:
        person_activity_checkpoint = pickle.load(f)

    with open(checkpoint_path, 'rb') as f:
        checkpoint_path = pickle.load(f)
    
    person_activity_cls = Person_Activity_Temporal_Classifer(
        num_classes=config.model['num_classes']['person_activity'],
        hidden_size=config.model['hyper_param']['person_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['person_activity']['num_layers']
    )
   
    person_activity_cls.load_state_dict(person_activity_checkpoint['model_state_dict'])    
   
    model = Group_Activity_Classifer(
        person_feature_extraction=person_act_cls, 
        num_classes=config.model['num_classes']['group_activity']
    )

    model.load_state_dict(checkpoint_path['model_state_dict'])    
   
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        collate_fn=group_collate_fn,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.ROOT}/modeling/baseline 5/outputs"
    prefix = "Group Activity Baseline 5 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"]['group_activity'])

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/configs/Baseline B5.yml"    
    PERSON_ACTIVITY_CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 5/outputs/Baseline_B5_Step_A_V1_2024_12_22_09_43/checkpoint_epoch_8.pkl"
    GROUP_ACTIVITY_CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 5/outputs/Baseline_B5_Step_B_V1_20241222_174053/checkpoint_epoch_31.pkl"
   
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
        num_classes=config.model['num_classes']['person_activity'],
        hidden_size=config.model['hyper_param']['person_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['person_activity']['num_layers']
    )

    model = Group_Activity_Classifer(
        person_feature_extraction=person_act_cls, 
        num_classes=config.model['num_classes']['group_activity']
    )

    summary(model)
    eval(args, PERSON_ACTIVITY_CHECKPOINT_PATH, GROUP_ACTIVITY_CHECKPOINT_PATH)
    # ==================================================
    # Group Activity Baseline 5 eval on testset
    # ==================================================
    # Accuracy : 77.04%
    # Average Loss: 0.6571
    # F1 Score (Weighted): 0.7707

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.89      0.76      0.82       192
    #      r_spike       0.92      0.93      0.93       173
    #       r-pass       0.69      0.67      0.68       210
    #   r_winpoint       0.45      0.43      0.44        87
    #   l_winpoint       0.51      0.56      0.54       102
    #       l-pass       0.73      0.83      0.78       226
    #      l-spike       0.90      0.92      0.91       179
    #        l_set       0.84      0.81      0.83       168

    #     accuracy                           0.77      1337
    #    macro avg       0.74      0.74      0.74      1337
    # weighted avg       0.77      0.77      0.77      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 5/outputs/Group_Activity_Baseline_5_eval_on_testset_confusion_matrix.png