import os
from argparse import ArgumentParser
from typing import Any

import numpy as np
import pandas as pd
import rootutils
import wandb
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.global_dicts import TASK_DICT  # noqa: E402

api = wandb.Api()


def convert_numpy_types(value: Any) -> Any:
    """Converts numpy types to native python types.

    :param value: Value to convert
    :return: Converted value
    """
    if isinstance(value, np.float64):  # type: ignore
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.int64):  # type: ignore
        return int(value)
    else:
        return value


def get_experiment(dataset: str) -> None:
    """Get experiment hyperparameters.

    :param dataset: Dataset to get the hyperparameters for
    """
    # Load the Hydra config
    config = OmegaConf.load(f"./configs/experiment/{dataset}.yaml")

    # Project is specified by <entity/project-name>
    runs = api.runs(f"prop_pred/{config.data.dataset}_v2")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})
    # Define the columns
    columns = [
        "config.model.optimizer.lr",
        "config.model.optimizer.weight_decay",
        "config.model.optimizer.momentum",
        "config.model.optimizer.rho",
        "config.model.optimizer.adaptive",
        "config.model.scheduler.T_0",
        "config.model.scheduler.T_mult",
        "config.model.net.hidden_dim1",
        "config.model.net.hidden_dim2",
        "config.model.net.hidden_dim3",
        "config.model.net.bottleneck_dim",
        "config.model.net.output_dim1",
        "config.model.net.output_dim2",
        "config.model.net.output_dim3",
        "config.model.net.dropout",
        "config.model.net.activation",
        "config.model.net.tie_weights",
        "config.model.recon_loss.alpha",
        "config.model.recon_loss.gamma",
        "config.model.loss_weights.recon_loss",
        "config.model.loss_weights.ubc_loss",
        "config.data.method",
        "config.data.descriptors",
        "summary.val/main_best",
    ]

    # Flatten the DataFrame
    flattened_df = pd.json_normalize(runs_df.to_dict(orient="records"))
    flattened_df = flattened_df[columns]

    # Group and aggregate the data
    grouped_df = (
        flattened_df.groupby(columns[:-1]).agg({"summary.val/main_best": "mean"}).reset_index()
    )

    # Create a new key in the config and assign a value to it
    methods = ["FG", "MFG", "FGR"]
    descriptors = [True, False]

    for method in methods:
        for descriptor in descriptors:
            hparams = (
                grouped_df[
                    (grouped_df["config.data.method"] == method)
                    & (grouped_df["config.data.descriptors"] == descriptor)
                ]
                .sort_values(
                    by="summary.val/main_best",
                    ascending=TASK_DICT[config.data.dataset][-1],
                )
                .iloc[0]
            )
            config.data.method = method
            config.data.descriptors = descriptor
            main_best_value = hparams["summary.val/main_best"]
            hparams = hparams.drop(
                [
                    "config.data.method",
                    "config.data.descriptors",
                    "summary.val/main_best",
                ]
            )
            for key, value in hparams.items():
                # Split the key into parts
                parts = key.split(".")  # type: ignore
                # Get the last part of the key
                last_part = parts.pop()
                # Get the dictionary to add the key and value to
                current_dict = config
                for part in parts[1:]:
                    # If the part is not in the dictionary, add it
                    if part not in current_dict:
                        current_dict[part] = {}

                    # Move to the next part of the dictionary
                    current_dict = current_dict[part]
                current_dict[last_part] = convert_numpy_types(value)

            os.makedirs(f"./configs/experiment/{dataset}", exist_ok=True)
            yaml_file = f"./configs/experiment/{dataset}/{method}_{descriptor}.yaml".lower()
            with open(yaml_file, "w") as f:
                f.write(
                    f"# @package _global_\n# summary.val/main_best: {main_best_value}\n"
                    + OmegaConf.to_yaml(config)
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    get_experiment(args.dataset)
