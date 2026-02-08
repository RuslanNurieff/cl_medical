import yaml

# Define your configuration
config = {
    'model': {
        'name': 'STFPM',
        'backbone': "resnet18",
        'layers': ['layer1', 'layer2', 'layer3'],
        'optimizer': None,
        'loss_function': None
    },
    'dataset': {
        'name': 'adnet',
        'path': '/mnt/disk2/VAD_DATASETS/Medical/Medical',
        "img_size": [256, 256],
        "mask_size": [256, 256],
        "transforms": None
    },
    'train': {
        'batch_size': 16,
        'epochs': 2,
        'early_stop': {
            "patience": 3,
            "min_delta": 0.02,
            "metric": "img_f1"
        },
        "device":"cuda:2",
        "logging": True,
        "aggregated": True,
        
    }
}

# Write to YAML file
with open('/home/ruslan/cl_medical/configs/STFPM.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)