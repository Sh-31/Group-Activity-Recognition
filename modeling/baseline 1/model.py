'''
Baseline B1-tuned Description :
--------------------------------
A straightforward image classifier based on ResNet-50 that is fine-tuned to 
classify group activities using a single frame from a video clip.
'''
import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

class Group_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifer, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
    
    def forward(self, x):

        return self.resnet50(x)


def model_summary(project_root, config_path):
    sys.path.append(os.path.abspath(project_root))

    from utils import load_config

    config = load_config(config_path)
    model = Group_Activity_Classifer(num_classes=config.model['num_classes'])

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

   



