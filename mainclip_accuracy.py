import json
import argparse
import torch
import wandb
from tqdm import tqdm
from torchvision import transforms

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
    resize_cifar10_img,
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
    rejection_class_index=-1,
    epsilon=None,
    verbose=False,
):
    if epsilon is None:
        epsilon = 8
    x_test, y_test = image, label
    attacks_to_run = (
        ["apgd-ce", "apgd-dlr"]
        if adaptive
        else ["apgd-ce-rejection", "apgd-dlr-rejection"]
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
    return torch.cat(clip_predictions).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Run experiments from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", default="cpu", type=str, help="Computation device")
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device(args.device)
    print(f"Running on {device}")
    use_wandb = False
    if use_wandb:
        wandb.init(project="multishield-experiments", config=config)

    for exp in config["experiments"]:
        set_seed(config["seed"])

        clip_model_id = exp["clip_model_id"]
        dataset = exp["dataset"]
        model_name = exp["model_name"]
        n_samples = exp["n_samples"]
        batch_size = exp["batch_size"]
        attack_parameters = exp["attack_parameters"]
        attack_parameters["device"] = device

        label_names = get_label_names(dataset)
        rejection_class = len(label_names)
        dataloaders = get_dataset_loaders(
            dataset, batch_size, n_samples, config["seed"]
        )
        # model = get_local_model(model_name, dataset).eval().to(device)
        clip_model_name, processor_name, tokenizer_name, use_open_clip = get_clip_model(
            clip_model_id
        )
        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
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
        # multi_shield = MultiShield(dnn=model, clip_model=clip_model, dataset=dataset)
        # # PRIMA ATTACCHIAMO DNN BASELINE DA SOLA
        # dnn_adv_results = run_attack(
        #     model, dataloaders["val"], partial(auto_attack, **attack_parameters)
        # )
        # # CREA UN DATALOADER ON GLI ADVERSARIAL EXAMPLES RITORNATI DALLA RUN_ATTACK
        # adv_loader_dnn = data_utils.DataLoader(
        #     data_utils.TensorDataset(
        #         dnn_adv_results["adv_examples"].clone().detach(),
        #         torch.tensor(dnn_adv_results["true_labels"]),
        #     ),
        #     batch_size=batch_size,
        #     shuffle=False,
        # )
        # # FA LA VALUTAZIONE DELLA DNN CON IL ADVERSARIAL DATALOADER SU DNN
        # dnn_acc = run_predictions(model, dataloaders["val"], adv_loader_dnn)
        # # FA LA VALUTAZIONE DI MULTISHIELD CON IL ADVERSARIAL DATALOADER SU DNN
        # ms_acc_on_dnn_adv = run_predictions(
        #     multi_shield, dataloaders["val"], adv_loader_dnn, rejection_class
        # )
        # attack_parameters.update(
        #     {
        #         "adaptive": True,
        #         "rejection_class_index": rejection_class,
        #     }
        # )
        # ms_adv_results = run_attack(
        #     multi_shield,
        #     dataloaders["val"],
        #     partial(
        #         auto_attack,
        #         **attack_parameters,
        #     ),
        # )
        # adv_loader_ms = data_utils.DataLoader(
        #     data_utils.TensorDataset(
        #         ms_adv_results["adv_examples"].clone().detach(),
        #         torch.tensor(ms_adv_results["true_labels"]),
        #     ),
        #     batch_size=batch_size,
        #     shuffle=False,
        # )
        # ms_acc = run_predictions(
        #     multi_shield, dataloaders["val"], adv_loader_ms, rejection_class
        # )
        clip_preds = [
            clip_model.clip_prediction(
                clip_model.create_img_emb(
                    resize_cifar10_img(img) if dataset == "cifar10" else img
                ),
                label,
            )
            for img, label in tqdm(dataloaders["val"], total=len(dataloaders["val"]))
        ]

        results = {
            # "DNN Clean Accuracy": dnn_acc["clean_accuracy"],
            # "DNN Robust Accuracy": dnn_acc["adv_accuracy"],
            # "MS Clean Accuracy": ms_acc["clean_accuracy"],
            # "MS Rejection Ratio": ms_acc["rejection_ratio_on_clean_samples"],
            # "MS Robust Accuracy (Non-Adaptive)": ms_acc_on_dnn_adv["adv_accuracy"],
            # "MS Rejection Ratio (Non-Adaptive)": ms_acc_on_dnn_adv[
            #     "rejection_ratio_on_adv_examples"
            # ],
            # "MS Robust Accuracy (Adaptive)": ms_acc["adv_accuracy"],
            # "MS Rejection Ratio (Adaptive)": ms_acc["rejection_ratio_on_adv_examples"],
            "CLIP Accuracy": compute_clip_accuracy(clip_preds),
        }

        print("\nResults:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        if use_wandb:
            wandb.log(results)


if __name__ == "__main__":
    main()
