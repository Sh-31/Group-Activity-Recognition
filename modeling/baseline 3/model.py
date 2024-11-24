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
import torchvision.models as models
from torchinfo import summary

class Person_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Person_Activity_Classifer, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.9),
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

        self.pool = nn.AdaptiveMaxPool2d((1, 4096))  # [12, 2048] -> [1, 4096]
        
        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        features = self.feature_extraction(x).squeeze() # [12, 2048, 1, 1] ->  [12, 2048]
        features = self.pool(features.unsqueeze(0)) # add batch dim [1, 12, 2048] to pool across all people
        return self.fc(features.squeeze(0)) # squeeze the extra batch dim [1 , 1, 4096] to [1 , 4096]


def model_summary(project_root, config_path):
    sys.path.append(os.path.abspath(project_root))

    from utils import load_config

    config = load_config(config_path)
    model = Person_Activity_Classifer(num_classes=config.model['num_classes'])

    summary(model)


if __name__ == "__main__":
    
    project_path = "/teamspace/studios/this_studio/Group-Activity-Recognition"

    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=project_path,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=f'{project_path}/modeling/configs/Baseline B1-tuned.yml',
                        help="Path to the YAML configuration file")

 
    args = parser.parse_args()
    model_summary(args.project_root, args.config_path)

   



