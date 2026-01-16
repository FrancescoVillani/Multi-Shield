import warnings
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()


def set_seed(seed: int) -> None:
    """
    Random seed generation for PyTorch. See https://pytorch.org/docs/stable/notes/randomness.html
        for further details.
    Args:
        seed (int): the seed for pseudonumber generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_attack(
    model: nn.Module,
    loader: DataLoader,
    attack,
    device: torch.device = None,
) -> dict:
    """
    Run adversarial attack on a model using the provided attack function.
    
    Code adapted from Official adversarial library repo:
    https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/utils/attack_utils.py
    
    Args:
        model: The model to attack
        loader: DataLoader with clean images
        attack: Attack function to generate adversarial examples
        device: Device to run on (auto-detected if None)
    
    Returns:
        Dictionary containing attack results and adversarial examples
    """
    model.eval()
    device = next(model.parameters()).device if device is None else device
    loader_length = len(loader)

    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
    else:
        start, end = 0, 0

    true_labels, predicted_labels_adv = [], []
    all_inputs, all_adv_examples = [], []
    linf_norms = []  # To store L∞ norm for each sample's perturbation

    for i, (inputs, labels) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        true_labels.append(labels.cpu().tolist())
        all_inputs.append(inputs)
        inputs, labels = inputs.to(device), labels.to(device)

        if device.type == "cuda":
            start.record()
            torch.cuda.reset_peak_memory_stats(device=device)
        
        try:
            adv_inputs = attack(model, inputs, labels)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "valid cuDNN" in str(e).lower():
                raise RuntimeError(f"Out of memory error during attack execution on batch {i}") from e
            raise

        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()

        if adv_inputs.min() < 0 or adv_inputs.max() > 1:
            warnings.warn(
                "Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1]."
            )
            adv_inputs.clamp_(min=0, max=1)

        # Calculate the L∞ norm of the perturbation for each sample
        perturbation = adv_inputs - inputs
        # Reshape to (batch_size, -1) and take the max absolute difference for each sample
        linf_norm_batch = perturbation.view(perturbation.size(0), -1).abs().max(dim=1)[0]
        linf_norms.append(linf_norm_batch.cpu().tolist())

        adv_logits = model(adv_inputs)
        adv_predictions = adv_logits.argmax(dim=1)
        predicted_labels_adv.append(adv_predictions.cpu().tolist())
        all_adv_examples.append(adv_inputs)

    data = {
        "true_labels": [item for sublist in true_labels for item in sublist],
        "pred_labels_adv": [item for sublist in predicted_labels_adv for item in sublist],
        "linf_norms": np.max([norm for sublist in linf_norms for norm in sublist])
    }

    if len(all_inputs) > 1:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_adv_examples = torch.cat(all_adv_examples, dim=0)
    data["inputs"] = all_inputs
    data["adv_examples"] = all_adv_examples

    return data


def run_predictions(
    model: nn.Module,
    clean_dataset: DataLoader,
    adv_dataset: DataLoader,
    rejection_class: int = None,
    device: torch.device = None,
) -> dict:
    """
    Evaluate model on clean and adversarial datasets.
    
    Args:
        model: Model to evaluate
        clean_dataset: DataLoader with clean images
        adv_dataset: DataLoader with adversarial images
        rejection_class: Index of rejection class (if using rejection)
        device: Device to run on (auto-detected if None)
    
    Returns:
        Dictionary containing accuracy metrics and predictions
    """
    model.eval()
    device = next(model.parameters()).device if device is None else device
    dataset_length = len(clean_dataset)

    rejections_on_clean_dataset, rejections_on_adv_dataset = [], []
    true_labels, predicted_labels, adv_predicted_labels = [], [], []

    for i, (inputs, labels) in enumerate(tqdm(clean_dataset, ncols=80, total=dataset_length)):
        true_labels.extend(labels.cpu().tolist())
        inputs, labels = inputs.to(device), labels.to(device)

        try:
            logits = model(inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "valid cuDNN" in str(e).lower():
                print(
                    "\n WARNING: ran out of memory, cannot perform experiments with this batch size"
                )
                raise e
            raise

        predictions = logits.argmax(dim=1)
        predicted_labels.extend(predictions.cpu().tolist())

        if rejection_class:
            rejections_on_clean_dataset.extend(
                (predictions == rejection_class).cpu().tolist()
            )

    for i, (adv_inputs, adv_labels) in enumerate(tqdm(adv_dataset, ncols=80, total=dataset_length)):
        adv_inputs = adv_inputs.to(device)

        try:
            adv_logits = model(adv_inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "valid cuDNN" in str(e).lower():
                print(
                    "\n WARNING: ran out of memory, cannot perform experiments with this batch size"
                )
                raise e
            raise

        adv_predictions = adv_logits.argmax(dim=1)
        adv_predicted_labels.extend(adv_predictions.cpu().tolist())

        if rejection_class:
            rejections_on_adv_dataset.extend(
                (adv_predictions == rejection_class).cpu().tolist()
            )

    if rejection_class:
        clean_accuracy = sum(
            [predicted_labels[i] == true_labels[i] for i in range(len(true_labels))]
        ) / len(true_labels)
        adv_accuracy = sum(
            [
                adv_predicted_labels[i] == true_labels[i]
                or adv_predicted_labels[i] == rejection_class
                for i in range(len(true_labels))
            ]
        ) / len(true_labels)
    else:
        clean_accuracy = sum(
            [predicted_labels[i] == true_labels[i] for i in range(len(true_labels))]
        ) / len(true_labels)
        adv_accuracy = sum(
            [adv_predicted_labels[i] == true_labels[i] for i in range(len(true_labels))]
        ) / len(true_labels)

    data = {
        "clean_accuracy": clean_accuracy,
        "adv_accuracy": adv_accuracy,
        "rejection_ratio_on_clean_samples": (
            sum(rejections_on_clean_dataset) / len(rejections_on_clean_dataset)
            if len(rejections_on_clean_dataset) > 0
            else 0
        ),
        "rejection_ratio_on_adv_examples": (
            sum(rejections_on_adv_dataset) / len(rejections_on_adv_dataset)
            if len(rejections_on_adv_dataset) > 0
            else 0
        ),
        "asr": 1 - adv_accuracy,
        "true_labels": true_labels,
        "pred_labels": predicted_labels,
        "adv_labels": adv_predicted_labels,
        "rejections_on_clean_dataset": rejections_on_clean_dataset,
        "rejections_on_adv_dataset": rejections_on_adv_dataset  
    }

    # Convert any tensors to lists
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].tolist()
    return data


def resize_to_224(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize a batch of images to 224x224 pixels. If the images have only one channel (e.g., MNIST),
    convert them to three channels by duplicating the channel.
    
    This function is fully differentiable and processes the entire batch at once for efficiency.

    Args:
        batch_tensor (torch.Tensor): A batch of images with shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: A batch of resized images with shape (batch_size, 3, 224, 224).
    """
    # Convert grayscale to RGB if needed (vectorized operation)
    if batch_tensor.size(1) == 1:
        batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
    
    # Resize entire batch at once (much faster than looping)
    resized_batch = F.interpolate(
        batch_tensor,
        size=(224, 224),
        mode='bilinear',
        align_corners=False,
    )
    
    return resized_batch


class GrayscaleToRGB(object):
    def __call__(self, image):
        if image.mode == 'L':
            image = image.convert("RGB")
        return image