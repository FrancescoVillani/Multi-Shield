import torch
import torchvision
import random
from ingredients.utilities import set_seed, GrayscaleToRGB
import torchvision.transforms as transforms
import os
import ssl
import json

torchvision.disable_beta_transforms_warning()
ssl._create_default_https_context = ssl._create_unverified_context

IMAGENET_TRAINING_PATH = "/data/datasets/imagenet"

CALTECH101_PATH = "./data/datasets"
CALTECH101_SPLIT_PATH = "./ingredients/dataset_preprocessing/caltech101_train_test_split.seed=1233.json"


def common_transform():
    return transforms.Compose([transforms.ToTensor()])


def get_label_names(dataset: str) -> list[str]:
    """
    Get class label names for a dataset.
    
    Args:
        dataset: Dataset name ('cifar10' or 'imagenet')
    
    Returns:
        List of class label names
    """
    labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
    with open(labels_path, 'r') as f:
        all_labels = json.load(f)
    
    if dataset not in all_labels:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(all_labels.keys())}")
    
    return all_labels[dataset]


def get_dataset_loader_cifar10(batch_size: int, n_examples: int):
    transform = transforms.Compose([transforms.ToTensor()])

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.CIFAR10(
        root="./data/datasets/", download=True, transform=transform
    )
    dataloaders["train"] = torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
    )

    image_datasets["val"] = torchvision.datasets.CIFAR10(
        root="./data/datasets/", train=False, download=True, transform=transform
    )
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")

    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(
            image_datasets["val"],
            random.sample(range(len(image_datasets["val"])), n_examples),
        )

    dataloaders["val"] = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
    )

    dataloaders["class_names"] = get_label_names("cifar10")

    torch.cuda.empty_cache()
    return dataloaders


def get_dataset_loader_imagenet(transform, batch_size, n_examples):
    train_path = os.path.join(IMAGENET_TRAINING_PATH, "val")

    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)

    imagenet_data_subset = torch.utils.data.Subset(
        imagenet_data, random.sample(range(len(imagenet_data)), n_examples)
    )

    resized_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    imagenet_data_subset.dataset.transform = resized_transform

    dum = imagenet_data_subset if n_examples > 0 else imagenet_data
    data_loader = {
        "val": torch.utils.data.DataLoader(
            dum, batch_size=batch_size, shuffle=False, num_workers=0
        )
    }

    data_loader["class_names"] = get_label_names("imagenet")

    return data_loader


def get_dataset_loader_caltech101(batch_size: int, n_examples: int):
    dataloaders = {}

    if os.path.exists(CALTECH101_SPLIT_PATH):
        with open(CALTECH101_SPLIT_PATH, "r") as f:
            split_indices = json.load(f)
            train_indices = split_indices["train"]
            test_indices = split_indices["test"]
    else:
        raise FileNotFoundError(f"Split not defined in path: {CALTECH101_SPLIT_PATH}")

    resized_transform = transforms.Compose([
        # transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        GrayscaleToRGB(),
        transforms.ToTensor()
    ])

    caltech101_data = torchvision.datasets.Caltech101(root=CALTECH101_PATH, transform=resized_transform, download=False)

    caltech101_train_data = torch.utils.data.Subset(caltech101_data, train_indices)

    if n_examples == 0:
        selected_test_indices = test_indices
    else:
        selected_test_indices = test_indices[:n_examples]

    caltech101_test_data = torch.utils.data.Subset(caltech101_data, selected_test_indices)

    dataloaders["train"] = torch.utils.data.DataLoader(caltech101_train_data, batch_size=batch_size, shuffle=True)
    dataloaders["val"] = torch.utils.data.DataLoader(caltech101_test_data, batch_size=batch_size, shuffle=False)

    dataloaders["class_names"] = get_label_names("caltech101")

    return dataloaders


def get_dataset_loader_stl10(batch_size: int, n_examples: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.STL10(
        root="./data/datasets/", split="train", download=True, transform=transform
    )
    dataloaders["train"] = torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
    )

    image_datasets["val"] = torchvision.datasets.STL10(
        root="./data/datasets/", split="test", download=True, transform=transform
    )
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")

    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(
            image_datasets["val"],
            random.sample(range(len(image_datasets["val"])), n_examples),
        )

    dataloaders["val"] = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
    )

    dataloaders["class_names"] = get_label_names("stl10")

    torch.cuda.empty_cache()
    return dataloaders


def get_dataset_loader_food101(batch_size: int, n_examples: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.Food101(
        root="./data/datasets/", split="train", download=True, transform=transform
    )
    dataloaders["train"] = torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
    )

    image_datasets["val"] = torchvision.datasets.Food101(
        root="./data/datasets/", split="test", download=True, transform=transform
    )
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")

    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(
            image_datasets["val"],
            random.sample(range(1, len(image_datasets["val"])), n_examples),
        )

    dataloaders["val"] = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
    )

    dataloaders["class_names"] = get_label_names("food101")

    torch.cuda.empty_cache()
    return dataloaders


def get_dataset_loader_pets(batch_size: int, n_examples: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.OxfordIIITPet(
        root="./data/datasets/", split="trainval", download=True, transform=transform
    )
    dataloaders["train"] = torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
    )

    image_datasets["val"] = torchvision.datasets.OxfordIIITPet(
        root="./data/datasets/", split="test", download=True, transform=transform
    )
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")

    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(
            image_datasets["val"],
            random.sample(range(1, len(image_datasets["val"])), n_examples),
        )

    dataloaders["val"] = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
    )

    dataloaders["class_names"] = get_label_names("pets")

    torch.cuda.empty_cache()
    return dataloaders


def get_dataset_loaders(dataset, batch_size, n_examples, seed, split_percentage=0.5):
    set_seed(seed=seed)
    transform = common_transform()

    loaders = {}

    if dataset == "cifar10":
        print(f"Loading CIFAR10 dataset with batch size {batch_size}")
        loaders = get_dataset_loader_cifar10(batch_size, n_examples)

    elif dataset == "imagenet":
        print(f"Loading IMAGENET dataset with batch size {batch_size}")
        loaders = get_dataset_loader_imagenet(transform, batch_size, n_examples)
    elif dataset == "caltech101":
        print(f"Loading CALTECH101 dataset with batch size {batch_size}")
        loaders = get_dataset_loader_caltech101(batch_size, n_examples)
    elif dataset == "stl10":
        print(f"Loading STL10 dataset with batch size {batch_size}")
        loaders = get_dataset_loader_stl10(batch_size, n_examples)
    elif dataset == "food101":
        print(f"Loading FOOD101 dataset with batch size {batch_size}")
        loaders = get_dataset_loader_food101(batch_size, n_examples)
    elif dataset == "pets":
        print(f"Loading PETS dataset with batch size {batch_size}")
        loaders = get_dataset_loader_pets(batch_size, n_examples)
    else:
        print("Please input a valid dataset (cifar10, imagenet, caltech101, stl10, food101, pets)")
        return loaders

    # Split the training set
    if "train" in loaders and split_percentage < 1:
        train_dataset = loaders["train"].dataset
        train_size = len(train_dataset)
        split_size = int(train_size * split_percentage)
        indices = list(range(train_size))
        random.seed(seed)
        random.shuffle(indices)
        train_indices, train2_indices = indices[:split_size], indices[split_size:]

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train2_subset = torch.utils.data.Subset(train_dataset, train2_indices)

        loaders["train"] = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        loaders["train2"] = torch.utils.data.DataLoader(train2_subset, batch_size=batch_size, shuffle=True, num_workers=2)


    return loaders