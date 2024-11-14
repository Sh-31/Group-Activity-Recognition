import os
import yaml
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


def save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, epoch_num):
    # Saving model as a pickle file at the end of each epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
    }
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch_num}.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at {checkpoint_path}")