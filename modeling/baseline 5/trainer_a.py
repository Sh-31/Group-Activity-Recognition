"""
Train person activity temporal classifer
"""
import os
import sys
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from datetime import datetime
import torch.multiprocessing as mp
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Person_Activity_Temporal_Classifer, person_collate_fn

ROOT = "/kaggle"
PROJECT_ROOT= "/kaggle/working/Group-Activity-Recognition"
CONFIG_FILE_PATH = f'{PROJECT_ROOT}/modeling/configs/Baseline B5.yml'

sys.path.append(os.path.abspath(PROJECT_ROOT))

from data_utils import Person_Activity_DataSet, person_activity_labels
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config, setup_logging, load_checkpoint, save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        predicted = outputs.argmax(1)
        target_class = targets.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(target_class).sum().item()
        
        if batch_idx % 100 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/BatchLoss', loss.item(), step)
            writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, step)
            
            log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
            logger.info(log_msg)
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
    writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate_model(model, val_loader, criterion, device, epoch, writer, logger, class_names):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            _, target_class = targets.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    f1_score = get_f1_score(y_true, y_pred, average="weighted")
    writer.add_scalar('Validation/F1Score', f1_score, epoch)
    
    fig = plot_confusion_matrix(y_true, y_pred, class_names)
    writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
    
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy, f1_score

def train_model(config_path, checkpoint_path=None):
   
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Person_Activity_Temporal_Classifer(
        num_classes=config.model['num_classes']['person_activity'],
        hidden_size=config.model['hyper_param']['person_activity']['hidden_size'],
        num_layers=config.model['hyper_param']['person_activity']['num_layers']
    )

    model = model.to(device)
    
    if config.training['person_activity']['optimizer'] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training['person_activity']['learning_rate'],
            weight_decay=config.training['person_activity']['weight_decay']
        )
    
    elif config.training['person_activity']['optimizer'] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training['person_activity']['learning_rate'],
            weight_decay=config.training['person_activity']['weight_decay']
        )   
    
    start_epoch = 0
    best_val_acc = 0
    update_optimizer = True

    if checkpoint_path:
        model, optimizer, loaded_config, exp_dir, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)
        exp_dir = "/kaggle/working/Group-Activity-Recognition/modeling/baseline 5/outputs/Baseline_B5_Step_A_V1_2024_12_22_09_43"
        logger = setup_logging(exp_dir)
       
        if loaded_config:
            # config = loaded_config
            logger.info(f"Resumed training from epoch {start_epoch}")

        if update_optimizer:
            if config.training['group_activity']['optimizer'] == "AdamW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.training['group_activity']['learning_rate'],
                    weight_decay=config.training['group_activity']['weight_decay']
                ) 
            
            elif config.training['group_activity']['optimizer'] == "SGD":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.training['group_activity']['learning_rate'],
                    momentum=config.training['group_activity']['momentum'],
                    weight_decay=config.training['group_activity']['weight_decay']
                )   
    else:
         timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
       
         exp_dir = os.path.join(
                f"{PROJECT_ROOT}/modeling/baseline 8/{config.experiment['output_dir']}",
                f"{config.experiment['name']}_V{config.experiment['version']}_{timestamp}"
            )
         
         os.makedirs(exp_dir, exist_ok=True)
         logger = setup_logging(exp_dir)

    logger.info(f"Starting experiment: {config.experiment['name']}_V{config.experiment['version']}")
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    logger.info(f"Using device: {device}")
    
    set_seed(config.experiment['seed'])
    logger.info(f"Set random seed: {config.experiment['seed']}")
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.55),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    train_dataset = Person_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['train'],
        labels=person_activity_labels,
        transform=train_transforms,
        seq=True, 
    )
    
    val_dataset = Person_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['validation'],
        labels=person_activity_labels,
        transform=val_transforms,
        seq=True, 
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['person_activity']['batch_size'],
        collate_fn=person_collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training['person_activity']['batch_size'],
        collate_fn=person_collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.training['person_activity']['label_smoothing']
    )
    
    scaler = GradScaler()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
    )
    
    config_save_path = os.path.join(exp_dir, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training["person_activity"]["epochs"]):
        
        logger.info(f'\nEpoch {epoch+1}/{config.training["person_activity"]["epochs"]}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger
        )
        
        val_loss, val_acc, val_f1_score = validate_model(
            model, val_loader, criterion, device, epoch, writer, logger, config.model['num_clases_label']['person_activity']
        )
        
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_acc:.2f}% | Valid F1 Score: {val_f1_score:.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, is_best=True)
        
        save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/LearningRate', current_lr, epoch)
        logger.info(f'Current learning rate: {current_lr}')
    
    writer.close()
    
    logger.info(f"Training completed.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    RESUME_CHECK_POINT  = f"{PROJECT_ROOT}/modeling/baseline 5/outputs/Baseline_B5_Step_A_V1_2024_12_22_09_43/checkpoint_epoch_9.pkl"
    train_model(CONFIG_FILE_PATH, RESUME_CHECK_POINT)
