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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Hierarchical_Group_Activity_Classifer, collate_fn

ROOT = "/kaggle"
PROJECT_ROOT = "/kaggle/working/Group-Activity-Recognition"
CHECK_POINT = "/kaggle/working/Group-Activity-Recognition/modeling/baseline 9 (end to end)/outputs/Baseline_B9_V1_2025_01_04_10_09/checkpoint_epoch_21.pkl"
CONFIG_FILE_PATH = f'{PROJECT_ROOT}/modeling/configs/Baseline B9.yml'

sys.path.append(os.path.abspath(PROJECT_ROOT))

from data_utils import Hierarchical_Group_Activity_DataSet, activities_labels
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger, rank):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, person_labels, group_labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        person_labels = person_labels.to(device)
        group_labels = group_labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss_1 = criterion(outputs['person_output'], person_labels)
            loss_2 = criterion(outputs['group_output'], group_labels)
            loss = loss_2 + (0.25 * loss_1)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Gather loss from all processes
        reduced_loss = loss.detach()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / dist.get_world_size()
        
        total_loss += reduced_loss.item()
        
        predicted = outputs['group_output'].argmax(1)
        target_class = group_labels.argmax(1)
        total += group_labels.size(0)
        correct += predicted.eq(target_class).sum().item()
        
        if batch_idx % 100 == 0 and batch_idx != 0:
            # Synchronize metrics across processes
            loss_tensor = torch.tensor(total_loss, device=device)
            correct_tensor = torch.tensor(correct, device=device)
            total_tensor = torch.tensor(total, device=device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            if rank == 0:
                avg_loss = loss_tensor.item() / (batch_idx + 1)
                accuracy = 100. * correct_tensor.item() / total_tensor.item()
                
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Training/BatchLoss', avg_loss, step)
                writer.add_scalar('Training/BatchAccuracy', accuracy, step)
                
                log_msg = (f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                          f'Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
                logger.info(log_msg)
        
        dist.barrier()
    
    loss_tensor = torch.tensor(total_loss, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    epoch_loss = loss_tensor.item() / len(train_loader)
    epoch_acc = 100. * correct_tensor.item() / total_tensor.item()
    
    if rank == 0:
        writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc
    
def validate_model(model, val_loader, criterion, device, epoch, writer, logger, class_names, rank):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, person_labels, group_labels in val_loader:
            inputs = inputs.to(device)
            person_labels = person_labels.to(device)
            group_labels = group_labels.to(device)
            
            outputs = model(inputs)
            loss_1 = criterion(outputs['person_output'], person_labels)
            loss_2 = criterion(outputs['group_output'], group_labels)
            
            loss = loss_2 + (0.25 * loss_1)
            
            total_loss += loss.item()
            
            _, predicted = outputs['group_output'].max(1)
            _, target_class = group_labels.max(1)
            total += group_labels.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    if dist.is_initialized():
        dist.all_reduce(torch.tensor(total_loss).to(device))
        dist.all_reduce(torch.tensor(total).to(device))
        dist.all_reduce(torch.tensor(correct).to(device))
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    if rank == 0:
        f1_score = get_f1_score(y_true, y_pred, average="weighted")
        writer.add_scalar('Validation/F1Score', f1_score, epoch)
        
        fig = plot_confusion_matrix(y_true, y_pred, class_names)
        writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
        
        writer.add_scalar('Validation/Loss', avg_loss, epoch)
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy, f1_score
    return avg_loss, accuracy, 0.0

def train_model_ddp(rank, world_size, config_path, checkpoint_path=None):
    setup_ddp(rank, world_size)
    config = load_config(config_path)
    device = torch.device(f"cuda:{rank}")
    
    model = Hierarchical_Group_Activity_Classifer(
        person_num_classes=config.model['num_classes']['person_activity'],
        group_num_classes=config.model['num_classes']['group_activity'],
        hidden_size=config.model['hyper_param']['hidden_size'],
        num_layers=config.model['hyper_param']['num_layers']
    ).to(device)

    # model = DDP(model, device_ids=[rank])

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
    
    start_epoch = 0
    best_val_acc = 0
    
    if checkpoint_path:
        model, optimizer, loaded_config, exp_dir, start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        
        model = DDP(model, device_ids=[rank])
    
        logger = setup_logging(exp_dir) if rank == 0 else None
    else:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        exp_dir = os.path.join(
            f"{PROJECT_ROOT}/modeling/baseline 9 (end to end)/{config.experiment['output_dir']}",
            f"{config.experiment['name']}_V{config.experiment['version']}_{timestamp}"
        )
        if rank == 0:
            os.makedirs(exp_dir, exist_ok=True)
            logger = setup_logging(exp_dir)
        else:
            logger = None
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard')) if rank == 0 else None
    
    if rank == 0:
        logger.info(f"Starting experiment: {config.experiment['name']}_V{config.experiment['version']}")
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Using optimizer: {config.training['optimizer']}, "
            f"lr: {config.training['learning_rate']}, "
            f"momentum: {config.training.get('momentum', 0)}, "
            f"weight_decay: {config.training['weight_decay']}")
    
    set_seed(config.experiment['seed'] + rank) # each GPU process has a different but deterministic random seed
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.90),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = Hierarchical_Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['train'],
        labels=activities_labels,
        transform=train_transforms,
    )
    
    val_dataset = Hierarchical_Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['validation'],
        labels=activities_labels,
        transform=val_transforms,
    )

    if rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training['batch_size'],
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training['label_smoothing'])
    scaler = GradScaler()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
    )
    
    if rank == 0:
        config_save_path = os.path.join(exp_dir, 'config.yml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.training["epochs"]):
        train_sampler.set_epoch(epoch)
      
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            writer, logger, rank
        )
        
        val_loss, val_acc, val_f1_score = validate_model(
            model, val_loader, criterion, device, epoch, writer, logger,
            config.model['num_clases_label']['group_activity'], rank
        )
        
        if rank == 0:
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_acc:.2f}% | Valid F1 Score: {val_f1_score:.4f}")
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model.module, optimizer, epoch, val_acc, config, exp_dir, is_best=True)
            
            save_checkpoint(model.module, optimizer, epoch, val_acc, config, exp_dir)
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/LearningRate', current_lr, epoch)
            logger.info(f'Current learning rate: {current_lr}')
    
    if writer is not None:
        writer.close()
    
    if rank == 0:
        logger.info(f"Training completed.")
    
    cleanup_ddp()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model_ddp,
        args=(world_size, CONFIG_FILE_PATH, CHECK_POINT),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True) # Use forkserver on Linux to avoid JAX conflicts 
    main()
