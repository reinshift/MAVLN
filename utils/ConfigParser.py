import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, d):
            self._dict_props = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    if all(isinstance(key, int) for key in v.keys()):
                        self._dict_props[k] = v
                    else:
                        setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
                    
        def __getattr__(self, name):
            if name in self._dict_props:
                return self._dict_props[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    return Config(config_dict)