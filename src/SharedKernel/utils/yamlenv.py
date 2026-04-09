import yaml

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    def __repr__(self):
        return str(self.__dict__) 

def load_env_yaml(path="config_env/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)

def load_redis_index(path="config_env/redis_index.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)
    return schema_dict
