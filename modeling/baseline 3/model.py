'''
Baseline 3 Description :
--------------------------------
fine-tuned on person-level action annotations (players).
Individual features are pooled across all people 
then use it train NN over group activity class.
'''
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
        self.in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)

class Group_Activity_ClassiferNN(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(Group_Activity_ClassiferNN, self).__init__()
        #  Extract feature layers of ResNet50 (excluding final FC layer)
        self.fc_in_features = person_feature_extraction.in_features
        self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-1])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [12, 2048] -> [1, 4096]
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        features = self.feature_extraction(x).squeeze() # [12, 2048, 1, 1] ->  [12, 2048]
        features = self.pool(features.unsqueeze(0)) # add batch dim [1, 12, 2048] to pool across all people
        return self.fc(features.squeeze(0)) # squeeze the extra batch dim [1 , 1, 4096] to [1 , 4096]


def model_summary(args):
    sys.path.append(os.path.abspath(args.project_root))

    from helper_utils import load_config

    config = load_config(args.config_path)
    person_act_cls = Person_Activity_Classifer(num_classes=config.model['num_classes']['person_activity'])
    model = Group_Activity_ClassiferNN(person_feature_extraction=person_act_cls, 
                                        num_classes=config.model['num_classes']['group_activity'])
    summary(model)

def eval(args, person_activity_checkpoint, checkpoint_path):

    sys.path.append(os.path.abspath(args.project_root))
    
    import pickle
    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels
    
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_activity_cls = Person_Activity_Classifer(num_classes=config.model['num_classes']['person_activity'])
   
    person_activity_cls = load_checkpoint(model=person_activity_cls, checkpoint_path=person_activity_checkpoint, device=device, optimizer=None)
    
    model = Group_Activity_ClassiferNN(person_feature_extraction=person_activity_cls, num_classes=config.model['num_classes']['group_activity'])

    model = load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device, optimizer=None)

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
        crops=True,
        seq=False, 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.project_root}/modeling/baseline 3/outputs"
    prefix = "Group Activity Baseline 2 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"]['group_activity'])

    return metrics


if __name__ == "__main__":
    
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/configs/Baseline B3_step_b.yml"
    PERSON_ACTIVITY_CHECKPOINT_PATH = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 3/outputs/Baseline_B3_step_A_V1_20241127_184841/checkpoint_epoch_0.pkl"
    CHECKPOINT_PATH = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 3/outputs/Baseline_B3_step_B_V1_20241127_192620/checkpoint_epoch_4.pkl"
   
    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    
    # model_summary(args) # Show model details 
    eval(args, PERSON_ACTIVITY_CHECKPOINT_PATH, CHECKPOINT_PATH) # eval model against  testset

    # ==================================================
    # Group Activity Baseline 2 eval on testset
    # ==================================================
    # Accuracy : 69.38%
    # Average Loss: 0.9217
    # F1 Score (Weighted): 0.6892

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.84      0.70      0.76      1728
    #      r_spike       0.83      0.82      0.82      1557
    #       r-pass       0.68      0.52      0.59      1890
    #   r_winpoint       0.49      0.13      0.21       783
    #   l_winpoint       0.37      0.86      0.51       918
    #       l-pass       0.70      0.66      0.68      2034
    #      l-spike       0.81      0.87      0.84      1611
    #        l_set       0.76      0.82      0.79      1512

    #     accuracy                           0.69     12033
    #    macro avg       0.69      0.67      0.65     12033
    # weighted avg       0.72      0.69      0.69     12033
