import yaml

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
