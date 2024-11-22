# import os
# import sys
# import yaml
# import torch
# import random
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.transforms import v2
# from datetime import datetime
# from torch.cuda.amp import autocast, GradScaler
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard.writer import SummaryWriter
# from model import Group_Activity_Classifer

# ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"
# sys.path.append(os.path.abspath(ROOT))

# from data_utils import Group_Activity_DataSet, group_activity_labels
# from eval_utils import get_f1_score , plot_confusion_matrix
# from utils import load_config, setup_logging, save_checkpoint

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         # inputs.shape : torch.Size([64, 3, 224, 224])
#         # targets.shape : torch.Size([64, 8])
#         optimizer.zero_grad()
        
#         with autocast(dtype=torch.float16):
#             outputs = model(inputs) # outputs.shape : torch.Size([64, 8])
#             loss = criterion(outputs, targets)
        
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         total_loss += loss.item()
        
#         predicted = outputs.argmax(1) 
#         target_class = targets.argmax(1)
#         total += targets.size(0)
#         correct += predicted.eq(target_class).sum().item()
        
#         if batch_idx % 10 == 0:
#             step = epoch * len(train_loader) + batch_idx
#             writer.add_scalar('Training/BatchLoss', loss.item(), step)
#             writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, step)
            
#             log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
#             logger.info(log_msg)
    
#     epoch_loss = total_loss / len(train_loader)
#     epoch_acc = 100. * correct / total
    
#     writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
#     writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)
    
#     return epoch_loss, epoch_acc

# def validate_model(model, val_loader, criterion, device, epoch, writer, logger, class_names):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     y_true = []
#     y_pred = []
    
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
#             total_loss += loss.item()
            
#             _, predicted = outputs.max(1)
#             _, target_class = targets.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(target_class).sum().item()
            
           
#             y_true.extend(target_class.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
    
#     avg_loss = total_loss / len(val_loader)
#     accuracy = 100. * correct / total
    
#     f1_score = get_f1_score(y_true, y_pred, average="weighted")
#     writer.add_scalar('Validation/F1Score', f1_score, epoch)
    
#     fig = plot_confusion_matrix(y_true, y_pred, class_names)
#     writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
    
#     writer.add_scalar('Validation/Loss', avg_loss, epoch)
#     writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
#     logger.info(f"Epoch {epoch} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f}")
    
#     return avg_loss, accuracy

# def train_model(config_path):
   
#     config = load_config(config_path)
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     exp_dir = os.path.join(
#         f"{ROOT}/modeling/baseline 1/{config.experiment['output_dir']}",
#         f"{config.experiment['name']}_V{config.experiment['version']}_{timestamp}"
#     )
#     os.makedirs(exp_dir, exist_ok=True)
    
#     logger = setup_logging(exp_dir)
#     logger.info(f"Starting experiment: {config.experiment['name']}_V{config.experiment['version']}")

#     writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")
    
#     set_seed(config.experiment['seed'])
#     logger.info(f"Set random seed: {config.experiment['seed']}")
    
#     train_transform = v2.Compose([
#         v2.ToPILImage(),
#         v2.Resize((224, 224)),
#         v2.ToImage(), 
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     val_transform = v2.Compose([
#         v2.ToPILImage(),
#         v2.Resize((224, 224)),
#         v2.ToImage(), 
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
    
#     train_dataset = Group_Activity_DataSet(
#         videos_path=f"{ROOT}/{config.data['videos_path']}",
#         annot_path=f"{ROOT}/{config.data['annot_path']}",
#         split=config.data['video_splits']['train'],
#         labels=group_activity_labels, 
#         transform=train_transform
#     )
    
#     val_dataset = Group_Activity_DataSet(
#         videos_path=f"{ROOT}/{config.data['videos_path']}",
#         annot_path=f"{ROOT}/{config.data['annot_path']}",
#         split=config.data['video_splits']['validation'],
#         labels=group_activity_labels,
#         transform=val_transform
#     )
    
#     logger.info(f"Training dataset size: {len(train_dataset)}")
#     logger.info(f"Validation dataset size: {len(val_dataset)}")
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.training['batch_size'],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.training['batch_size'],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     model = Group_Activity_Classifer(num_classes=config.model['num_classes'])
#     model = model.to(device)
#     logger.info(f"Model initialized: {config.model['name']}")
    
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config.training['learning_rate'],
#         weight_decay=config.training['weight_decay']
#     )
    
#     criterion = nn.CrossEntropyLoss()
#     scaler = GradScaler()
    
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.1,
#         patience=5,
#         verbose=True
#     )
    
#     config_save_path = os.path.join(exp_dir, 'config.yml')
#     with open(config_save_path, 'w') as f:
#         yaml.dump(config, f)
#     logger.info(f"Configuration saved to: {config_save_path}")
    
#     logger.info("Starting training...")
#     for epoch in range(config.training['epochs']):
#         logger.info(f'\nEpoch {epoch+1}/{config.training["epochs"]}')
        
#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger
#         )
        
#         val_loss, val_acc = validate_model(model, val_loader, criterion, device, epoch, writer, logger, config.model['num_clases_label'])
#         scheduler.step(val_loss)
        
#         current_lr = optimizer.param_groups[0]['lr']
#         writer.add_scalar('Training/LearningRate', current_lr, epoch)
#         logger.info(f'Current learning rate: {current_lr}')
#         save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, epoch)
       
#     writer.close()
    
#     final_model_path = os.path.join(exp_dir, 'final_model.pth')
#     torch.save({
#         'epoch': config.training['epochs'],
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_acc': val_acc,
#         'config': config,
#     }, final_model_path)
    
#     logger.info(f"Training completed. Final model saved to: {final_model_path}")

# if __name__ == "__main__":
#     config_path = os.path.join(ROOT, "modeling/configs/Baseline B1-tuned.yml")
#     train_model(config_path)
#     # tensorboard --logdir="/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 1/outputs/Baseline_B1_tuned_V1_20241117_044805/tensorboard"