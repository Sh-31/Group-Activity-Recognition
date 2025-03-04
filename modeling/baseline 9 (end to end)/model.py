"""
Baseline 9 Description:
In previous  baselines, person activity and group activity losses were handled independently, resulting in a two-stage model. 
Baseline 9 combines these processes into a unified, end-to-end training pipeline. This approach allows for the simultaneous 
optimization of both person-level and group-level activity classification using a shared gradient flow.
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class Hierarchical_Group_Activity_Classifer(nn.Module):
    def __init__(self, person_num_classes, group_num_classes, hidden_size, num_layers):
        super(Hierarchical_Group_Activity_Classifer, self).__init__()

        self.feature_extractor = nn.Sequential(
            *list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-1]
        )
        self.layer_norm_1 = nn.LayerNorm(512)
        
        self.gru_1 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, person_num_classes)
        )

        self.layer_norm_2 = nn.LayerNorm(512)
        self.pool = nn.AdaptiveMaxPool2d((1, 256))
     
        self.gru_2 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, group_num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape  # seq => frames
        x = x.view(b*bb*seq, c, h, w)  # (b * bb * seq, c, h, w)
        x1 = self.feature_extractor(x) # (batch * bbox * seq, 512, 1 , 1)

        x1 = x1.view(b*bb, seq, -1)       # (batch * bbox, seq, 512)
        x1 = self.layer_norm_1(x1)          # (batch * bbox, seq, 512)
        x2, (h_1 , c_1) = self.gru_1(x1) # (batch * bbox, seq, hidden_size)

        y1 = self.fc_1(x2[:, -1, :])  # (batch, person_num_classes)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet34 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene (same as paper).
       
        x = x.view(b*seq, bb, -1) # (batch * seq, bbox, hidden_size)
        team_1 = x[:, :6, :]      # (batch * seq, 6, hidden_size)
        team_2 = x[:, 6:, :]      # (batch * seq, 6, hidden_size)

        team_1 = self.pool(team_1) # (batch * seq, 1, 256)
        team_2 = self.pool(team_2) # (batch * seq, 1, 256)
        x = torch.cat([team_1, team_2], dim=1)  # (batch * seq, 2, 256)
       
        x = x.view(b, seq, -1) # (batch, seq, 512)
        x = self.layer_norm_2(x) # (batch, seq, 512)
        x, (h_2 , c_2) = self.gru_2(x) # (batch, seq, hidden_size)

        x = x[:, -1, :]     # (batch, hidden_size)
        y2 = self.fc_2(x)   # (batch, group_num_classes)
        return {'person_output': y1, 'group_output': y2}


def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, person_labels, group_labels  = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_person_labels = []

    for clip, label in zip(clips, person_labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_person_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_person_labels = torch.stack(padded_person_labels)
    group_labels = torch.stack(group_labels)
    
    group_labels = group_labels[:,-1, :] # # utils the label of last frame
    padded_person_labels = padded_person_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_person_labels.shape # batch, bbox, num_clases
    padded_person_labels = padded_person_labels.view(b*bb, num_class)

    return padded_clips, padded_person_labels, group_labels

def get_sampler_weights(dataset):
    labels = []
    for idx in range(len(dataset)):
        _, person_label, group_label = dataset[idx]
        labels.append(group_label[-1].argmax().item()) # take one label of the 9 frame 
    
    class_counts = torch.bincount(torch.tensor(labels))

    num_samples = len(labels)
    class_weights = (1.0 / class_counts.float()) 
    class_weights = class_weights / class_weights.sum()
    
    samples_weights = [class_weights[label] for label in labels]
    samples_weights = torch.DoubleTensor(samples_weights)
    
    return samples_weights, class_weights

def eval(args, checkpoint_path):

    sys.path.append(os.path.abspath(args.ROOT))
    from helper_utils import load_config, load_checkpoint
    from eval_utils import plot_confusion_matrix
    from data_utils import Hierarchical_Group_Activity_DataSet, activities_labels
    
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Hierarchical_Group_Activity_Classifer(
        person_num_classes=config.model['num_classes']['person_activity'],
        group_num_classes=config.model['num_classes']['group_activity'],
        hidden_size=config.model['hyper_param']['hidden_size'],
        num_layers=config.model['hyper_param']['num_layers']
    ).to(device)

   
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
    
    test_dataset = Hierarchical_Group_Activity_DataSet(
        videos_path=f"{args.ROOT}/{config.data['videos_path']}",
        annot_path=f"{args.ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=activities_labels,
        transform=test_transforms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{args.ROOT}/modeling/baseline 9 (end to end)/outputs"
    prefix = "Group Activity Baseline 9 eval on testset"

    model.eval()  
    y_true = []
    y_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, person_labels, group_labels in test_loader:
            inputs = inputs.to(device)
            person_labels = person_labels.to(device)
            group_labels = group_labels.to(device)
            
            outputs = model(inputs)
            loss_1 = criterion(outputs['person_output'], person_labels)
            loss_2 = criterion(outputs['group_output'], group_labels)
            
            loss = (0.70 * loss_2) + (0.30 * loss_1)
            
            total_loss += loss.item()
            
            _, predicted = outputs['group_output'].max(1)
            _, target_class = group_labels.max(1)
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report_dict = classification_report(y_true, y_pred, target_names=config.model["num_clases_label"]["group_activity"], output_dict=True)
    if isinstance(report_dict, dict):
        accuracy = report_dict["accuracy"] * 100
 
    avg_loss = total_loss / len(test_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print(f"{prefix}")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.2f}%")
    if criterion:
        print(f"Average Loss: {avg_loss:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=config.model["num_clases_label"]["group_activity"]))

    if config.model["num_clases_label"]["group_activity"]:
        save_path = f"{path}/{prefix.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=config.model["num_clases_label"]["group_activity"], save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics

if __name__ == "__main__":

    ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition" 
    MODEL_CONFIG = f"{ROOT}/modeling/configs/Baseline B9 (end to end).yml"   
    CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 9 (end to end)/outputs/Baseline_B9_V1_2025_02_22_10_36/checkpoint_epoch_49.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=MODEL_CONFIG,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()
    sys.path.append(os.path.abspath(args.ROOT))

    from helper_utils import load_config
    config = load_config(args.config_path)

    model = Hierarchical_Group_Activity_Classifer(
        person_num_classes=config.model['num_classes']['person_activity'],
        group_num_classes=config.model['num_classes']['group_activity'],
        hidden_size=config.model['hyper_param']['hidden_size'],
        num_layers=config.model['hyper_param']['num_layers']
    )

    summary(model)
    eval(args, CHECKPOINT_PATH)
    # ==================================================
    # Group Activity Baseline 9 eval on testset
    # ==================================================
    # Accuracy : 93.12%
    # Average Loss: 0.4360
    # F1 Score (Weighted): 0.9311

    # Classification Report:
    #               precision    recall  f1-score   support

    #        r_set       0.94      0.86      0.90       192
    #      r_spike       0.93      0.93      0.93       173
    #       r-pass       0.89      0.95      0.92       210
    #   r_winpoint       0.91      0.99      0.95        87
    #   l_winpoint       0.96      0.94      0.95       102
    #       l-pass       0.95      0.94      0.94       226
    #      l-spike       0.96      0.93      0.95       179
    #        l_set       0.91      0.93      0.92       168

    #     accuracy                           0.93      1337
    #    macro avg       0.93      0.94      0.93      1337
    # weighted avg       0.93      0.93      0.93      1337

    # Confusion matrix saved to /teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 9 (end to end)/outputs/Group_Activity_Baseline_9_eval_on_testset_confusion_matrix.png