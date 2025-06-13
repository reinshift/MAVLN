# MAVLN

Use Mantis as base model, to generate multiple actions for every agents simultaneously.

## Quick Usage
Before running, you have to download Mantis model to make the files like:
```
.
├── configs
├── Mantis 
│   ├── mantis-model # download Mantis model here
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── model-00002-of-00004.safetensors
│   │   ├── model-00003-of-00004.safetensors
|   |   ├── ...
├── ...
```

> during training process we used `swanlab` to track experiment logs, if you would like not to record these