# Pretrained Models Directory

This directory is for storing custom trained model weights.

## Structure

``` json
pretrained_models/
└── custom/
    ├── resnet50_cifar10.pth
    ├── wide_resnet50_2_cifar10.pth
    ├── resnet50_caltech101.pth
    ├── resnet50_stl10.pth
    ├── resnet50_food101.pth
    ├── resnet50_pets.pth
    └── ... (other custom models)
```

## Usage

For CIFAR-10 and ImageNet, Multi-Shield uses robust models from [RobustBench](https://robustbench.github.io/) which are automatically downloaded.

For other datasets (Caltech-101, STL-10, Food-101, Oxford-IIIT Pets), you need to:

1. Train your own models using PyTorch
2. Save the model state dict to this directory
3. Update the paths in `ingredients/models.py` if you use different filenames

## Training Custom Models

Example training script structure:

```python
import torch
import torchvision

# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)

# Modify final layer for your dataset
num_classes = 101  # e.g., for Caltech-101
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Train your model
# ... training code ...

# Save the model
torch.save(model.state_dict(), 'pretrained_models/custom/resnet50_caltech101.pth')
```

## Note

The placeholder paths in `ingredients/models.py` point to files in `pretrained_models/custom/`. These files are **not included** in the repository - you must train and provide your own models if you want to use datasets beyond CIFAR-10 and ImageNet.
