"""
Baseline 3 Description :
--------------------------------
fine-tuned on person-level action annotations (players).
Individual features are pooled across all people 
then use it train NN over group activity class.
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
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [12, 2048] -> [1, 2048]
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        b, bb, c, h, w = x.shape # batch, bbox, channals, hight, width
        x = x.view(b*bb, c, h, w) # [b*bb, c, h, w]
        x = self.feature_extraction(x) # [b*bb, 2048, 1, 1] 

        x = x.view(b, bb, -1) # (b, bb, 2048)
        x = self.pool(x) # [b, 1, 2048] 
        
        x = x.squeeze(dim=1) # [b, 2048]
        x = self.fc(x) # [b, num_classes] 
        return x 

def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame.
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3)))
            clip = torch.cat((clip, clip_padding), dim=0)
            
        padded_clips.append(clip)
        padded_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)
    
    return padded_clips, padded_labels

def eval(args, checkpoint_path):

    sys.path.append(os.path.abspath(args.ROOT))
    
    from helper_utils import load_config, load_checkpoint
    from eval_utils import model_eval
    from data_utils import Group_Activity_DataSet, group_activity_labels
    
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_activity_cls = Person_Activity_Classifer(
        num_classes=config.model['num_classes']['person_activity']
    )
   
    model = Group_Activity_ClassiferNN(
        person_feature_extraction=person_activity_cls, 
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
        seq=False, 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=40,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()

    metrics = model_eval(
        model=model, 
        data_loader=test_loader, 
        criterion=criterion, 
        device=device,
        path=f"{args.ROOT}/modeling/baseline 3/outputs", 
        prefix="Group Activity Baseline 3 eval on testset", 
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/configs/Baseline B3_step_b.yml"
    CHECKPOINT_PATH = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 3/outputs/Baseline_B3_step_B_V1_20250107_001804/checkpoint_epoch_8.pkl"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    
    sys.path.append(os.path.abspath(args.ROOT))
    from helper_utils import load_config

    config = load_config(args.config_path)
    person_act_cls = Person_Activity_Classifer(num_classes=config.model['num_classes']['person_activity'])
    model = Group_Activity_ClassiferNN(person_feature_extraction=person_act_cls, 
                                        num_classes=config.model['num_classes']['group_activity'])
    
    summary(model) # Show model details 
    eval(args, CHECKPOINT_PATH) # eval model against  testset
   
    #==================================================
    # Group Activity Baseline 3 eval on testset
    # ==================================================
    # Accuracy : 80.25%
    # Average Loss: 0.5982
    # F1 Score (Weighted): 0.8024

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.86      0.78      0.82      1728
    #      r_spike       0.90      0.88      0.89      1557
    #       r-pass       0.74      0.84      0.78      1890
    #   r_winpoint       0.55      0.52      0.53       783
    #   l_winpoint       0.63      0.62      0.63       918
    #       l-pass       0.82      0.82      0.82      2034
    #      l-spike       0.90      0.89      0.90      1611
    #        l_set       0.84      0.84      0.84      1512

    #     accuracy                           0.80     12033
    #    macro avg       0.78      0.77      0.78     12033
    # weighted avg       0.80      0.80      0.80     12033

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 3/outputs/Group_Activity_Baseline_3_eval_on_testset_confusion_matrix.png