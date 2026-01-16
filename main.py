"""
Multi-Shield: Robust Image Classification with Multi-Modal Large Language Models

This script runs the Multi-Shield evaluation experiments. It evaluates:
1. Standalone DNN classifier accuracy (clean and adversarial)
2. Multi-Shield accuracy against non-adaptive attacks
3. Multi-Shield accuracy against adaptive attacks

Usage:
    python main.py --config=configs/config_cifar10.json --device=cuda --seed=1233
"""

import json
import argparse
import torch
from tqdm import tqdm
from torchvision import transforms
import os
from datetime import datetime
from functools import partial

import torch.utils.data as data_utils
from ingredients.models import get_clip_model, get_local_model
from ingredients.dataset import get_dataset_loaders, get_label_names
from models.MultiShield import MultiShield
from models.CLIP import ClipModel
from attacks.modified_autoattack import AutoAttack
from ingredients.utilities import (
    run_attack,
    run_predictions,
    set_seed,
    resize_to_224,
)
import torchvision

torchvision.disable_beta_transforms_warning()


def read_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)


def auto_attack(
    model,
    image,
    label,
    device,
    adaptive=False,
    rejection_class_index=None,
    epsilon=None,
    verbose=False,
):
    """
    Run AutoAttack against the given model.
    
    Args:
        model: The model to attack
        image: Input images
        label: True labels
        device: Computation device
        adaptive: If True, use Multi-Shield specific attack variants
        rejection_class_index: Index of the rejection class (for adaptive attacks)
        epsilon: Perturbation budget (default: 8/255 for L-inf)
        verbose: Print attack progress
    
    Returns:
        Adversarial examples
    """
    if epsilon is None:
        epsilon = 8
    x_test, y_test = image, label
    attacks_to_run = (
        ["apgd-ce-rejection-ms", "apgd-dlr-rejection-ms"]
        if adaptive
        else ["apgd-ce", "apgd-dlr"]
    )

    adversary = AutoAttack(
        model,
        rejection_class_index=rejection_class_index,
        norm="Linf",
        eps=epsilon / 255,
        version="custom",
        attacks_to_run=attacks_to_run,
        verbose=verbose,
        device=device,
    )
    adversary.apgd.n_restarts = 1
    adversarial_examples = adversary.run_standard_evaluation(x_test, y_test)
    return adversarial_examples


def compute_clip_accuracy(clip_predictions):
    """Compute accuracy from CLIP predictions."""
    return torch.cat(clip_predictions).mean().item()


def initialize_experiment(exp, config, device, seed):
    """
    Initialize experiment components: models, dataloaders, and Multi-Shield.
    
    Returns:
        Dictionary containing all experiment components
    """
    set_seed(seed)

    clip_model_id = exp["clip_model_id"]
    dataset = exp["dataset"]
    model_name = exp["model_name"]
    n_samples = exp["n_samples"]
    batch_size = exp["batch_size"]
    attack_parameters = exp["attack_parameters"]
    attack_parameters["device"] = device

    images_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    label_names = get_label_names(dataset)
    rejection_class = len(label_names)
    dataloaders = get_dataset_loaders(dataset, batch_size, n_samples, seed)
    model = get_local_model(model_name, dataset, images_normalize).eval().to(device)
    
    clip_model_name, processor_name, tokenizer_name, use_open_clip = get_clip_model(
        clip_model_id
    )
    clip_model = ClipModel(
        clip_model_name,
        processor_name,
        tokenizer_name,
        use_open_clip,
        label_names,
        torch_preprocess=images_normalize,
        dataset=dataset,
        device=device,
        resize=partial(resize_to_224) if dataset in ["mnist", "cifar10"] else None
    )
    multi_shield = MultiShield(dnn=model, clip_model=clip_model)

    return {
        "dataloaders": dataloaders,
        "model": model,
        "clip_model": clip_model,
        "multi_shield": multi_shield,
        "attack_parameters": attack_parameters,
        "rejection_class": rejection_class,
        "batch_size": batch_size,
        "dataset": dataset,
        "device": device,
    }


def perform_attack(model, dataloaders, attack_parameters, batch_size):
    """Run attack and return results with adversarial dataloader."""
    dnn_adv_results = run_attack(
        model, dataloaders["val"], partial(auto_attack, **attack_parameters)
    )
    
    adv_loader_dnn = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.as_tensor(dnn_adv_results["adv_examples"]),
            torch.tensor(dnn_adv_results["true_labels"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return dnn_adv_results, adv_loader_dnn


def evaluate_model(model, dataloaders, adv_loader, rejection_class=None):
    """Evaluate model on clean and adversarial examples."""
    return run_predictions(model, dataloaders["val"], adv_loader, rejection_class)


def run_experiment(exp, config, device, seed):
    """
    Run a complete Multi-Shield evaluation experiment.
    
    Steps:
    1. Attack standalone DNN classifier
    2. Evaluate DNN and Multi-Shield on DNN adversarial examples
    3. Attack Multi-Shield with adaptive attack
    4. Evaluate Multi-Shield on adaptive adversarial examples
    5. Evaluate CLIP accuracy
    6. Save results
    """
    experiment = initialize_experiment(exp, config, device, seed)

    print("Performing the attack against the classifier standalone")
    dnn_adv_results, adv_loader_dnn = perform_attack(
        experiment["model"], 
        experiment["dataloaders"], 
        experiment["attack_parameters"], 
        experiment["batch_size"]
    )

    print("Evaluating the standalone classifier")
    dnn_acc = evaluate_model(
        experiment["model"], experiment["dataloaders"], adv_loader_dnn
    )

    print("Evaluating Multi-Shield against non-adaptive attack")
    ms_acc_on_dnn_adv = evaluate_model(
        experiment["multi_shield"], 
        experiment["dataloaders"], 
        adv_loader_dnn, 
        experiment["rejection_class"]
    )

    # Configure adaptive attack
    experiment["attack_parameters"].update({
        "adaptive": True,
        "rejection_class_index": experiment["rejection_class"],
    })

    print("Running adaptive attack on Multi-Shield")
    ms_adv_results, adv_loader_ms = perform_attack(
        experiment["multi_shield"], 
        experiment["dataloaders"], 
        experiment["attack_parameters"], 
        experiment["batch_size"]
    )

    print("Evaluating Multi-Shield against the adaptive attack")
    ms_acc = evaluate_model(
        experiment["multi_shield"], 
        experiment["dataloaders"], 
        adv_loader_ms, 
        experiment["rejection_class"]
    )

    # Evaluate CLIP accuracy
    clip_predictions = [
        experiment["clip_model"].clip_prediction(
            experiment["clip_model"].create_image_embedding(image.to(device)),
            label,
        )
        for image, label in tqdm(
            experiment["dataloaders"]["val"], 
            total=len(experiment["dataloaders"]["val"]),
            desc="Evaluating CLIP"
        )
    ]

    results = {
        "DNN Clean Accuracy": dnn_acc["clean_accuracy"],
        "DNN Robust Accuracy": dnn_acc["adv_accuracy"],
        "MS Clean Accuracy": ms_acc["clean_accuracy"],
        "MS Rejection Ratio": ms_acc["rejection_ratio_on_clean_samples"],
        "MS Robust Accuracy (Non-Adaptive)": ms_acc_on_dnn_adv["adv_accuracy"],
        "MS Rejection Ratio (Non-Adaptive)": ms_acc_on_dnn_adv["rejection_ratio_on_adv_examples"],
        "MS Robust Accuracy (Adaptive)": ms_acc["adv_accuracy"],
        "MS Rejection Ratio (Adaptive)": ms_acc["rejection_ratio_on_adv_examples"],
        "CLIP Accuracy": compute_clip_accuracy(clip_predictions),
        "linf_normale": dnn_adv_results["linf_norms"],
        "linf_adaptive": ms_adv_results["linf_norms"],
    }

    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Save results
    results_dir = os.path.join(
        "results", exp["dataset"], exp["model_name"], exp["clip_model_id"]
    )
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        results_dir, 
        f"MS_seed{seed}_samples{exp['n_samples']}_time{timestamp}"
    )
    
    with open(f"{results_file}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save config (remove device before saving)
    exp["attack_parameters"].pop("device")
    with open(f"{results_file}config.json", "w") as f:
        json.dump(exp, f, indent=4)


def main():
    """Main entry point for Multi-Shield experiments."""
    parser = argparse.ArgumentParser(
        description="Run Multi-Shield experiments from a config file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Computation device (cpu or cuda)"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device(args.device)
    print(f"Running on {device}")

    for exp in config["experiments"]:
        run_experiment(exp, config, device, seed=args.seed)


if __name__ == "__main__":
    main()
