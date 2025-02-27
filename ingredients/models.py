from functools import partial
import torch
from torch import nn
from robustbench import load_model

import torchvision

class WrapperModelNormalize(nn.Module):
    def __init__(self, model, torch_preprocess):
        super().__init__()
        self.model = model
        self.torch_preprocess = torch_preprocess

    def forward(self, x):
        x = self.torch_preprocess(x)
        return self.model(x)

carmon_2019 = {
    'name': 'Carmon2019Unlabeled',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
augustin_2020 = {
    'name': 'Augustin2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}

standard = {
    'name': 'Standard',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
gowal_2021 = {
    'name': 'Gowal2021Improving_70_16_ddpm_100m',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
chen_2020 = {
    'name': 'Chen2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

xu_2023 = {
    'name': 'Xu2023Exploring_WRN-28-10',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

addepalli2022 = {
    'name': 'Addepalli2022Efficient_RN18',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}


def load_robustbench_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model, model_dir="./checkpoints")
    return model


_local_cifar_models = {
    'carmon2019': partial(load_robustbench_model, name=carmon_2019['name'], dataset=carmon_2019['dataset'],
                          threat_model=carmon_2019['threat_model']),
    'augustin2020': partial(load_robustbench_model, name=augustin_2020['name'], dataset=augustin_2020['dataset'],
                            threat_model=augustin_2020['threat_model']),
    'standard': partial(load_robustbench_model, name=standard['name'], dataset=standard['dataset'],
                        threat_model=standard['threat_model']),
    'gowal2021': partial(load_robustbench_model, name=gowal_2021['name'], dataset=gowal_2021['dataset'],
                         threat_model=gowal_2021['threat_model']),
    'chen2020': partial(load_robustbench_model, name=chen_2020['name'], dataset=chen_2020['dataset'],
                        threat_model=chen_2020['threat_model']),
    'xu2023': partial(load_robustbench_model, name=xu_2023['name'], dataset=xu_2023['dataset'],
                      threat_model=xu_2023['threat_model']),
    'addepalli2022': partial(load_robustbench_model, name=addepalli2022['name'], dataset=addepalli2022['dataset'],
                             threat_model=addepalli2022['threat_model']),
}


def get_local_cifar_model(name: str, dataset: str) -> nn.Module:
    return _local_cifar_models[name]()


debenedetti2020light_s = {
    'name': 'Debenedetti2022Light_XCiT-S12',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

singh2023_convnext_L = {
    'name': 'Singh2023Revisiting_ConvNeXt-L-ConvStem',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

singh2023_Vit_B = {
    'name': 'Singh2023Revisiting_ViT-B-ConvStem',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

singh2023_ConvNeXt_B = {
    'name': 'Singh2023Revisiting_ConvNeXt-B-ConvStem',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

liu2023 = {
    'name': 'Liu2023Comprehensive_Swin-B',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

_local_imagenet_models = {
    'debenedetti2020light_s': partial(load_robustbench_model, name=debenedetti2020light_s['name'],
                                      dataset=debenedetti2020light_s['dataset'],
                                      threat_model=debenedetti2020light_s['threat_model']),
    'singh2023_convnext_L': partial(load_robustbench_model, name=singh2023_convnext_L['name'],
                                    dataset=singh2023_convnext_L['dataset'],
                                    threat_model=singh2023_convnext_L['threat_model']),
    'singh2023_Vit_B': partial(load_robustbench_model, name=singh2023_Vit_B['name'],
                               dataset=singh2023_Vit_B['dataset'],
                               threat_model=singh2023_Vit_B['threat_model']),
    'singh2023_ConvNeXt_B': partial(load_robustbench_model, name=singh2023_ConvNeXt_B['name'],
                                    dataset=singh2023_ConvNeXt_B['dataset'],
                                    threat_model=singh2023_ConvNeXt_B['threat_model']),
    'liu2023': partial(load_robustbench_model, name=liu2023['name'],
                       dataset=liu2023['dataset'],
                       threat_model=liu2023['threat_model'])
}


def load_local_model(path: str, model_type: str, new_num_classes: int = None, torch_preprocess: torchvision.transforms.Normalize = None) -> nn.Module:
    model = model_type(pretrained=True)
    if new_num_classes is not None:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, new_num_classes)

    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=False)

    if torch_preprocess is not None:
        original_forward = model.forward

        def preprocess_forward(x, *args, **kwargs):
            x = torch_preprocess(x)
            return original_forward(x, *args, **kwargs)
        
        model.forward = preprocess_forward

    return model.eval()


caltech101_resnet18 = {
    'path': "./pretrained_models/resnet18.model_seed=1123.dataset=caltech101.dataset_seed=1233.pth",
    'model_type': torchvision.models.resnet18,
    'new_num_classes': 101
}


_local_caltech101_models = {
    "caltech101_resnet18": partial(load_local_model, path=caltech101_resnet18["path"],
                                   model_type=caltech101_resnet18["model_type"],
                                   new_num_classes=caltech101_resnet18["new_num_classes"])
}


stl10_resnet18 = {
    'path': "./pretrained_models/resnet18.model_seed=1123.dataset=stl10.dataset_seed=1233.pth",
    'model_type': torchvision.models.resnet18,
    'new_num_classes': 10
}


_local_stl10_models = {
    "stl10_resnet18": partial(load_local_model, path=stl10_resnet18["path"],
                                   model_type=stl10_resnet18["model_type"],
                                   new_num_classes=stl10_resnet18["new_num_classes"])
}


def get_local_model(name: str, dataset: str, torch_preprocess: torchvision.transforms.Normalize = None) -> nn.Module:
    print(f"Loading {name}")
    if dataset == 'cifar10':
        return _local_cifar_models[name]()
    elif dataset == 'imagenet':
        return _local_imagenet_models[name]()
    elif dataset == 'caltech101':
        return _local_caltech101_models[name](torch_preprocess=torch_preprocess)
    elif dataset == 'stl10':
        return _local_stl10_models[name](torch_preprocess=torch_preprocess)


class ClipModelConfig:
    def __init__(self, model, processor, tokenizer, use_open_clip):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.use_open_clip = use_open_clip

    def get_model(self):
        return self.model, self.processor, self.tokenizer, self.use_open_clip


tangake_finetuned_config = ClipModelConfig(
    model='tanganke/clip-vit-base-patch32_cifar10',
    processor='openai/clip-vit-base-patch32',
    tokenizer=None,
    use_open_clip=False
)

clipa_ViT_L = ClipModelConfig(
    model='hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B',
    processor=None,
    tokenizer="hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B",
    use_open_clip=True
)

clip_models_dict = {
    'tangake_finetuned': tangake_finetuned_config.get_model,
    'clipa_ViT_L': clipa_ViT_L.get_model
}


def get_clip_model(config_name):
    config_func = clip_models_dict.get(config_name)
    if config_func:
        return config_func()
    else:
        raise ValueError(f"Model configuration '{config_name}' not found.")
