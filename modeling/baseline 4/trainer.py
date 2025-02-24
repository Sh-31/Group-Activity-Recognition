import os
import sys
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Group_Activity_Temporal_Classifer, collate_fn, FocalLoss

ROOT = "/kaggle/"
PROJECT_ROOT= "/kaggle/working/Group-Activity-Recognition"
CONFIG_FILE_PATH = "/kaggle/working/Group-Activity-Recognition/modeling/configs/Baseline B4.yml"

sys.path.append(os.path.abspath(PROJECT_ROOT))

from data_utils import Group_Activity_DataSet, group_activity_labels
from eval_utils import get_f1_score , plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint

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
        
        if batch_idx % 10 == 0 and batch_idx != 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/BatchLoss', loss.item(), step)
            writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, step)
            
            log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
            logger.info(log_msg)
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    
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
            
            predicted = outputs.argmax(1) 
            target_class = targets.argmax(1)
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
    
    logger.info(f"Epoch {epoch} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f}")
    
    return avg_loss, accuracy

def train_model(config_path):
   
    config = load_config(config_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(
        f"{PROJECT_ROOT}/modeling/baseline 4/{config.experiment['output_dir']}",
        f"{config.experiment['name']}_V{config.experiment['version']}_{timestamp}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = setup_logging(exp_dir)
    logger.info(f"Starting experiment: {config.experiment['name']}_V{config.experiment['version']}")

    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    set_seed(config.experiment['seed'])
    logger.info(f"Set random seed: {config.experiment['seed']}")
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise(),
            A.MotionBlur(blur_limit=5), 
            A.MedianBlur(blur_limit=5)  
        ], p=0.95),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90()
        ], p=0.10),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    train_dataset = Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['train'],
        crops=False,
        seq=True,
        labels=group_activity_labels, 
        transform=train_transforms
    )
    
    val_dataset = Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['validation'],
        crops=False,
        seq=True,
        labels=group_activity_labels,
        transform=val_transforms
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size']['train'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training['batch_size']['val'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    model = Group_Activity_Temporal_Classifer(
        num_classes=config.model['num_classes'],
        input_size=config.model['input_size'],
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers']
        )

    model = model.to(device)
    
    if config.training['optimizer'] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=config.training['weight_decay']
        )
    elif config.training['optimizer'] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=config.training['weight_decay']
        )     
    
    total_samples = len(train_dataset)
    labels = [label.argmax().item() for batch in train_loader for label in batch[1]]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()  

    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.training['label_smoothing'],
        weight=class_weights
    )
    
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )
    
    config_save_path = os.path.join(exp_dir, 'config.yml')

    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    logger.info("Starting training...")
    for epoch in range(config.training['epochs']):
        logger.info(f'\nEpoch {epoch+1}/{config.training["epochs"]}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger
        )
        
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, device, epoch, writer, logger, config.model['num_clases_label']
        )
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/LearningRate', current_lr, epoch)
        logger.info(f'Current learning rate: {current_lr}')
        save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir)
       
    writer.close()
    
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save({
        'epoch': config.training['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to: {final_model_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_model(CONFIG_FILE_PATH)
    # tensorboard --logdir="/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 1/outputs/Baseline_B1_tuned_V1_20241117_044805/tensorboard"
