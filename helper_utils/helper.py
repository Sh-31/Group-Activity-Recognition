import os
import yaml
import torch
import pickle

class Config:
    def __init__(self, config_dict):
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.data = config_dict.get("data", {})
        self.experiment = config_dict.get("experiment", {})
    
    def __repr__(self):
        return f"Config(model={self.model}, training={self.training}, data={self.data}, experiment={self.experiment})"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)    
    return config

def save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, is_best=False):
    """
    Save model checkpoint
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        val_acc: Validation accuracy
        config: Training configuration
        exp_dir: Experiment directory path
        epoch_num: Epoch number for filename
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'exp_dir':exp_dir,
        'config': config,
    }
    
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch}.pkl")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
 
    if is_best:
        best_model_path = os.path.join(exp_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved at {best_model_path}")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load model and training state from checkpoint
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer
        device: Device to load the model on
    Returns:
        tuple: (model, optimizer, config, exp_dir, start_epoch)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            torch.save(checkpoint, checkpoint_path)    
        except Exception as pickle_error:
            raise RuntimeError(
                f"Failed to load checkpoint using both torch.load and pickle.load: {pickle_error}"
            ) from pickle_error

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer == None: return model
       
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer states to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    start_epoch = checkpoint['epoch'] + 1
    config = checkpoint.get('config', None)
    exp_dir = checkpoint.get('exp_dir', None)
    
    return model, optimizer, config, exp_dir, start_epoch
        
