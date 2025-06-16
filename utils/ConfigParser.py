import yaml

class Config:
    def __init__(self, d):
        self._dict_props = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if any(isinstance(key, int) for key in v.keys()) or k == 'action_map':
                    self._dict_props[k] = v
                else:
                    setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
                
    def __getattr__(self, name):
        if hasattr(self, '_dict_props') and name in self._dict_props:
            return self._dict_props[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __repr__(self):
        attrs = {}
        for k, v in self.__dict__.items():
            if k != '_dict_props':
                attrs[k] = v
        
        if hasattr(self, '_dict_props'):
            for k, v in self._dict_props.items():
                attrs[k] = v
                
        return f"Config({attrs})"

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)