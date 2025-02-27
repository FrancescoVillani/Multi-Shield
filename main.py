import json
import argparse
import torch
import wandb
from tqdm import tqdm
from torchvision import transforms
import os  # Add this import
from datetime import datetime  # Add this import

from functools import partial
import torch.utils.data as data_utils
from ingredients.models import get_clip_model, get_local_model
from ingredients.dataset import get_dataset_loaders, get_label_names
from models.MultiShield import MultiShield
from models.CLIP import ClipModel
from modified_autoattack import AutoAttack
from ingredients.utilities import (
    run_attack,
    run_predictions,
    set_seed,
    resize_image_224,
)
import torchvision

torchvision.disable_beta_transforms_warning()


def read_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)


def auto_attack(
    model,
    image,
    label,
    device,
    adaptive=False,
    rejection_class_index=0,
    epsilon=None,
    verbose=False,
):
    if epsilon is None:
        epsilon = 8
    x_test, y_test = image, label
    attacks_to_run = (
        ["apgd-ce-rejection", "apgd-dlr-rejection"]

        if adaptive
        else ["apgd-ce", "apgd-dlr"]
    )

    print(f"Epsilon is: {epsilon}, so {epsilon/255}")
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
    return torch.cat(clip_predictions).mean().item()


def initialize_experiment(exp, config, device):
    set_seed(config["seed"])
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(project="multishield-experiments", config=config)

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
    dataloaders = get_dataset_loaders(
        dataset, batch_size, n_samples, config["seed"]
    )
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
    )
    multi_shield = MultiShield(dnn=model, clip_model=clip_model)

    return {
        "use_wandb": use_wandb,
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
    dnn_adv_results = run_attack(
        model, dataloaders["val"], partial(auto_attack, **attack_parameters)
    )
    adv_loader_dnn = data_utils.DataLoader(
        data_utils.TensorDataset(
            dnn_adv_results["adv_examples"].clone().detach(),
            torch.tensor(dnn_adv_results["true_labels"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return dnn_adv_results, adv_loader_dnn


def evaluate_model(model, dataloaders, adv_loader, rejection_class=None):
    return run_predictions(model, dataloaders["val"], adv_loader, rejection_class)


def run_experiment(exp, config, device):
    experiment = initialize_experiment(exp, config, device)

    print("Performing the attack against the classified standalone")
    dnn_adv_results, adv_loader_dnn = perform_attack(
        experiment["model"], experiment["dataloaders"], experiment["attack_parameters"], experiment["batch_size"]
    )

    print("Evaluating the standalone classifier")
    dnn_acc = evaluate_model(experiment["model"], experiment["dataloaders"], adv_loader_dnn)

    print("Evaluating Multi-Shield")
    ms_acc_on_dnn_adv = evaluate_model(
        experiment["multi_shield"], experiment["dataloaders"], adv_loader_dnn, experiment["rejection_class"]
    )

    experiment["attack_parameters"].update(
        {
            "adaptive": True,
            "rejection_class_index": experiment["rejection_class"],
        }
    )

    print("Running adaptive attack on Multi-Shield")
    ms_adv_results, adv_loader_ms = perform_attack(
        experiment["multi_shield"], experiment["dataloaders"], experiment["attack_parameters"], experiment["batch_size"]
    )

    print("Evaluating Multi-Shield against the adaptive attack")
    ms_acc = evaluate_model(
        experiment["multi_shield"], experiment["dataloaders"], adv_loader_ms, experiment["rejection_class"]
    )

    clip_preds = [
        experiment["clip_model"].clip_prediction(
            experiment["clip_model"].create_img_emb(
                resize_image_224(img.to(device)) if experiment["dataset"] == "cifar10" else img
            ),
            label,
        )
        for img, label in tqdm(experiment["dataloaders"]["val"], total=len(experiment["dataloaders"]["val"]))
    ]

    results = {
        "DNN Clean Accuracy": dnn_acc["clean_accuracy"],
        "DNN Robust Accuracy": dnn_acc["adv_accuracy"],
        "MS Clean Accuracy": ms_acc["clean_accuracy"],
        "MS Rejection Ratio": ms_acc["rejection_ratio_on_clean_samples"],
        "MS Robust Accuracy (Non-Adaptive)": ms_acc_on_dnn_adv["adv_accuracy"],
        "MS Rejection Ratio (Non-Adaptive)": ms_acc_on_dnn_adv[
            "rejection_ratio_on_adv_examples"
        ],
        "MS Robust Accuracy (Adaptive)": ms_acc["adv_accuracy"],
        "MS Rejection Ratio (Adaptive)": ms_acc["rejection_ratio_on_adv_examples"],
        "CLIP Accuracy": compute_clip_accuracy(clip_preds),
        "linf_normale": dnn_adv_results["linf_norms"],
        "linf_adaptive": ms_adv_results["linf_norms"],
    }

    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    if experiment["use_wandb"]:
        wandb.log(results)
        wandb.run.name = f"{exp['model_name']}_{exp['dataset']}_{exp['clip_model_id']}_{exp['n_samples']}"
        wandb.run.save()

    # Create results folder structure
    results_dir = os.path.join("results", exp["dataset"], exp["model_name"], exp["clip_model_id"])
    os.makedirs(results_dir, exist_ok=True)
    # Include date and time in the results file name to avoid duplicates
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"seed{config['seed']}samples{exp['n_samples']}time{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run experiments from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", default="cpu", type=str, help="Computation device")
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device(args.device)
    print(f"Running on {device}")

    for exp in config["experiments"]:
        run_experiment(exp, config, device)


if __name__ == "__main__":
    main()
