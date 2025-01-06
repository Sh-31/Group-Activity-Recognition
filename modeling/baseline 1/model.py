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
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary

class Group_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifer, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.resnet50(x)


def model_summary(args):
    sys.path.append(os.path.abspath(args.project_root))

    from helper_utils import load_config

    config = load_config(args.config_path)
    model = Group_Activity_Classifer(num_classes=config.model['num_classes'])

    summary(model)


def eval(args, checkpoint_path):
    sys.path.append(os.path.abspath(args.project_root))

    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels

    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Group_Activity_Classifer(num_classes=config.model['num_classes'])
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
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.project_root}/modeling/baseline 1/outputs"
    prefix = "Group Activity Baseline 1 eval on testset"

    metrics = model_eval(model=model, data_loader=test_loader, criterion=criterion, device=device , path=path, prefix=prefix, class_names=config.model["num_clases_label"])

    return metrics


if __name__ == "__main__":
    
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/configs/Baseline B1-tuned.yml"
    CHECKPOINT_PATH = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 1/outputs/Baseline_B1_tuned_V1_20250106_212915/checkpoint_epoch_7.pkl"

    parser = argparse.ArgumentParser(description="Group Activity Recognition Model Configuration")
    parser.add_argument("--project_root", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()

    # model_summary(args) # Show model details 
    eval(args, CHECKPOINT_PATH) # eval model against  testset

    # ==================================================
    # Group Activity Baseline 1 eval on testset
    # ==================================================
    # Accuracy : 72.66%
    # Average Loss: 1.1451
    # F1 Score (Weighted): 0.7263

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.70      0.66      0.68      1728
    #      r_spike       0.76      0.77      0.77      1557
    #       r-pass       0.63      0.65      0.64      1890
    #   r_winpoint       0.82      0.78      0.80       783
    #   l_winpoint       0.83      0.88      0.85       918
    #       l-pass       0.66      0.67      0.67      2034
    #      l-spike       0.83      0.84      0.84      1611
    #        l_set       0.70      0.68      0.69      1512

    #     accuracy                           0.73     12033
    #    macro avg       0.74      0.74      0.74     12033
    # weighted avg       0.73      0.73      0.73     12033

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 1/outputs/Group_Activity_Baseline_1_eval_on_testset_confusion_matrix.png

    

   



