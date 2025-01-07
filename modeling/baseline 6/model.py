"""
Baseline 6 Description :
--------------------------------
fine-tuned on person-level action annotations (players).
Individual features are pooled across all people 
then try to learn temporal features on image level
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

class Person_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Person_Activity_Classifer, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)

class Group_Activity_Classifer_Temporal(nn.Module):
    def __init__(self, person_feature_extraction, hidden_size, num_classes):
        super(Group_Activity_Classifer_Temporal, self).__init__()
   
        self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-1])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [12, 2048] -> [1, 2048]
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            batch_first=True
        ) 

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        b, bb, seq , c, h, w = x.shape # batch, bbox, seq, channals, hight, width
        x = x.view(b*bb*seq, c, h, w) # [b*bb*seq, c, h, w]
        x = self.feature_extraction(x) # [b*bb*seq, 2048, 1, 1] 

        x = x.view(b*seq, bb, -1) # (b*seq, bb, 2048)
        x = self.pool(x) # [b*seq, 1, 2048] 
        
        x = x.squeeze(dim=1) # [b*seq, 2048]
        x = x.view(b, seq, -1) # [b, seq, 2048]

        x, (h, c) = self.lstm(x) # [b, seq, hidden]
        x = x [:, -1, :] # [b, hidden] 

        x = self.fc(x) # [b, num_classes] 
        return x 

def collate_fn(batch):
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

    person_act_cls = Person_Activity_Classifer(
        num_classes=config.model['num_classes']['person_activity'],
    )
   
    model = Group_Activity_Classifer_Temporal(
        person_feature_extraction=person_act_cls, 
        hidden_size=config.model['hyper_param']['group_activity']['hidden_size'],
        num_classes=config.model['num_classes']['group_activity']
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
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.ROOT}/modeling/baseline 6/outputs"
    prefix = "Group Activity Baseline 6 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"]['group_activity'])

    return metrics

if __name__  == "__main__" :
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = f"{ROOT}/modeling/configs/Baseline B6.yml"   
    GROUP_ACTIVITY_CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 6/outputs/Baseline_B6_V1_20250107_033216/checkpoint_epoch_13.pkl"
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    sys.path.append(os.path.abspath(args.ROOT))

    from helper_utils import load_config
    config = load_config(args.config_path)

    person_act_cls = Person_Activity_Classifer(
        num_classes=config.model['num_classes']['person_activity'],
    )

    model = Group_Activity_Classifer_Temporal(
        person_feature_extraction=person_act_cls, 
        hidden_size=config.model['hyper_param']['group_activity']['hidden_size'],
        num_classes=config.model['num_classes']['group_activity']
    )

    summary(model)
    eval(args, GROUP_ACTIVITY_CHECKPOINT_PATH)
    # ==================================================
    # Group Activity Baseline 6 eval on testset
    # ==================================================
    # Accuracy : 84.52%
    # Average Loss: 0.4303
    # F1 Score (Weighted): 0.8399

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.90      0.84      0.87       192
    #      r_spike       0.93      0.82      0.87       173
    #       r-pass       0.81      0.93      0.87       210
    #   r_winpoint       0.63      0.28      0.38        87
    #   l_winpoint       0.58      0.88      0.70       102
    #       l-pass       0.92      0.92      0.92       226
    #      l-spike       0.86      0.91      0.88       179
    #        l_set       0.94      0.89      0.91       168

    #     accuracy                           0.85      1337
    #    macro avg       0.82      0.81      0.80      1337
    # weighted avg       0.85      0.85      0.84      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 6/outputs/Group_Activity_Baseline_6_eval_on_testset_confusion_matrix.png