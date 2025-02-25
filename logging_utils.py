import csv
import os


def creating_csv(
    dnn_clean_acc,
    dnn_robust_acc,
    ms_clean_acc,
    ms_adv_acc_on_dnn_adv_exmp,
    ms_adv_acc,
    rejection_ratio_ms_on_clean_dataset,
    rejection_ratio_ms_on_dnn_adv_dataset,
    rejection_ratio_ms_on_ms_adv_dataset,
    model_name,
    n_samples,
    clip_accuracy,
):
    result_file = f"results_{model_name}_on_{n_samples}_samples.csv"

    final_result = os.path.join("results", result_file)

    first_row = [
        f"{model_name} Clean Accuracy",
        f"{model_name} Robust Accuracy",
        "CLIP Clean Accuracy",
        "Multi-Shield Clean Accuracy",
        "Multi-Shield Rejection Ratio on Clean Dataset",
    ]

    first_row_values = [
        dnn_clean_acc,
        dnn_robust_acc,
        clip_accuracy,
        ms_clean_acc,
        rejection_ratio_ms_on_clean_dataset,
    ]

    second_row = [
        "Multi-Shield Robust Accuracy",
        "Multi-Shield Rejection Ratio on Adversarial Dataset",
        "Multi-Shield Robust Accuracy under Adaptive Attack",
        "Multi-Shield Rejection Ratio on Adversarial Dataset (under Adaptive Attack)",
    ]

    second_row_values = [
        ms_adv_acc_on_dnn_adv_exmp,
        rejection_ratio_ms_on_dnn_adv_dataset,
        ms_adv_acc,
        rejection_ratio_ms_on_ms_adv_dataset,
    ]

    with open(final_result, mode="w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
        writer.writerow(first_row_values)
        writer.writerow(second_row)
        writer.writerow(second_row_values)
