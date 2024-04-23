import glob
import os
import pickle
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import scienceplots  # noqa: F401
import seaborn as sns
import torch
from captum.attr import (
    Attribution,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
)
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
from torch import nn
from torchmetrics.functional.clustering import davies_bouldin_score
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data.components.global_dicts import TASK_DICT  # noqa: E402
from src.data.components.utils import (  # noqa: E402
    get_descriptors,
    smiles2vector_fg,
    smiles2vector_mfg,
)
from src.models.fgr_module import FGRLitModule  # noqa: E402

# Set matplotlib style
plt.style.use(["science", "nature", "vibrant"])
# Set color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class WrapperModel(nn.Module):
    """Wrapper model to extract the latent space from the model.

    :param nn: PyTorch model
    """

    def __init__(self, model):
        """Constructor method.

        :param model: PyTorch model
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor
        :param desc: Descriptor tensor
        :return: Latent space tensor
        """
        return self.model((x, desc))[0]


class FigureGenerator:
    """Class for generating figures."""

    def __init__(
        self,
        data_dir: str = "./data",
        dataset: str = "BACE",
        mode: str = "input",
        descriptor: bool = True,
    ):
        """Initialize 'FigGen'.

        :param data_dir: Directory of data
        :param dataset: Dataset name
        :param mode: Mode for generating figures
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.mode = mode
        self.descriptor = descriptor
        self.dataset_dir = self.data_dir / "reports" / "figures" / dataset
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.fg_df = pd.read_parquet(self.data_dir / "processed" / "training" / "fg.parquet")
        self.fgroups = self.fg_df["SMARTS"].tolist()
        self.fg_names = self.fg_df["Name"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in self.fgroups]  # Convert to RDKit Mol
        self.tokenizer = Tokenizer.from_file(
            str(
                self.data_dir / "processed" / "training" / "tokenizers" / f"BPE_pubchem_{500}.json"
            )
        )  # Load tokenizer
        self.mfg_names = list(self.tokenizer.get_vocab().keys())
        self.descriptor_names = [x[0] for x in Descriptors._descList]
        self.data_df = pd.read_parquet(
            self.data_dir / "processed" / "tasks" / dataset / f"{dataset}.parquet"
        )
        self.labels = self.data_df.drop(columns=["SMILES"])
        self.smiles = self.data_df["SMILES"].tolist()
        self.input_names = self.fg_names + self.mfg_names + self.descriptor_names
        self.input_names = [
            name.replace("_", r"\_")
            .replace("#", r"\#")
            .replace("%", r"\%")
            .replace("β", r"$\beta$")
            for name in self.input_names
        ]

    def get_representation(self, smiles: List[str], method: str) -> np.ndarray:
        """Get representation of molecules.

        :param smiles: List of SMILES
        :param method: Method of representation
        :raises ValueError: Method not supported
        :raises ValueError: Mode not supported
        :return: Representation of molecules
        """
        if method == "FG":
            x = np.stack([smiles2vector_fg(x, self.fgroups_list) for x in smiles])
        elif method == "MFG":
            x = np.stack([smiles2vector_mfg(x, self.tokenizer) for x in smiles])
        elif method == "FGR":
            f_g = np.stack([smiles2vector_fg(x, self.fgroups_list) for x in smiles])
            mfg = np.stack([smiles2vector_mfg(x, self.tokenizer) for x in smiles])
            x = np.concatenate((f_g, mfg), axis=1)  # Concatenate both vectors
        else:
            raise ValueError("Method not supported")  # Raise error if method not supported
        if self.mode == "input":
            return x
        elif self.mode == "train":
            ckpt_paths = glob.glob(
                f"./models/{self.dataset}/{method}/scaffold/{self.descriptor}/checkpoints/*/epoch_*.ckpt"
            )
            x = torch.tensor(x, dtype=torch.float32, device="cpu")
            desc = torch.tensor(
                np.concatenate([get_descriptors(smi) for smi in smiles], axis=0),
                dtype=torch.float32,
                device="cpu",
            )
            z_d_values = []
            for ckpt_path in ckpt_paths:
                model = FGRLitModule.load_from_checkpoint(ckpt_path).to("cpu")
                model.eval()
                with torch.no_grad():
                    if self.descriptor:
                        z_d = model((x, desc))
                    else:
                        z_d = model(x)
                z_d_values.append(z_d[1].cpu().numpy())
            average_z_d = np.asarray(z_d_values).mean(axis=0)
            return average_z_d
        else:
            raise ValueError("Mode not supported")

    def get_scaffolds(self) -> Tuple[pd.DataFrame, LabelEncoder]:
        """Get scaffolds of molecules.

        :return: Scaffolds of molecules and label encoder
        """
        scaffolds = defaultdict(set)
        idx2mol = dict(zip(list(range(len(self.smiles))), self.smiles))
        error_smiles = 0
        for i, smi in enumerate(self.smiles):
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=MolFromSmiles(smi), includeChirality=False
                )
                scaffolds[scaffold].add(i)
            except BaseException:
                print(smi + " returns RDKit error and is thus omitted...")
                error_smiles += 1

        top_5_scaffolds = sorted(
            ((k, v) for k, v in scaffolds.items() if k != ""),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:5]
        data = [
            (idx2mol[idx], scaffold) for scaffold, indices in top_5_scaffolds for idx in indices
        ]
        scaffold_df = pd.DataFrame(data, columns=["SMILES", "Label"])
        label_encoder = LabelEncoder()
        scaffold_df["Label"] = label_encoder.fit_transform(scaffold_df["Label"])
        return scaffold_df, label_encoder

    def plot_scaffolds(self, label_encoder: LabelEncoder) -> None:
        """Plot scaffolds.

        :param label_encoder: Label encoder
        """
        scaffold_dir = f"{self.dataset_dir}/scaffolds"
        os.makedirs(scaffold_dir, exist_ok=True)
        for i in range(5):
            scaffold = label_encoder.inverse_transform([i])[0]
            mol = MolFromSmiles(scaffold)
            Draw.MolToFile(
                mol,
                f"{scaffold_dir}/scaffold_{i}.svg",
            )

    def plot_kde_2d(self, method: str, components: np.ndarray, ax: mpl.axes._axes.Axes) -> None:  # type: ignore
        """Plot 2D KDE.

        :param method: Method of representation
        :param components: 2D components
        :param ax: Matplotlib axes
        """
        sns.kdeplot(
            x=components[:, 0],
            y=components[:, 1],
            cmap="rocket_r",
            fill=True,
            levels=500,
            bw_adjust=0.2,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
        ax.set_title(r"$\textbf{method:}$" f"{method}")

    def plot_kde(self, components: np.ndarray, ax: mpl.axes._axes.Axes) -> None:  # type: ignore
        """Plot 1D KDE.

        :param components: 2D components
        :param ax: Matplotlib axes
        """
        # Calculate the angles
        angles = np.arctan2(components[:, 1], components[:, 0])
        sns.kdeplot(x=angles, cmap="Blues", fill=True, ax=ax)
        ax.set_xlabel("Angles")
        ax.set_ylabel("Density")

    def plot_tsne(
        self,
        label_input: str,
        method: str,
        dbi: float,
        components: np.ndarray,
        labels: np.ndarray,
        ax: mpl.axes._axes.Axes,  # type: ignore
    ) -> None:
        """Plot t-SNE.

        :param label_input: Label input type
        :param method: Method of representation
        :param dbi: Davies-Bouldin index
        :param components: 2D components
        :param labels: Labels
        :param ax: Matplotlib axes
        """
        if TASK_DICT[self.dataset][-1] and label_input != "Scaffold":
            sns.scatterplot(
                ax=ax,
                x=components[:, 0],
                y=components[:, 1],
                hue=labels,
                palette=sns.color_palette("coolwarm", as_cmap=True),
            )
        else:
            sns.scatterplot(
                ax=ax,
                x=components[:, 0],
                y=components[:, 1],
                hue=labels,
                palette=sns.color_palette(colors),
            )
            ax.text(
                0.95,
                0.05,
                r"$\textbf{DBI:}$" f"{dbi}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="grey", alpha=0.2, edgecolor="black"),
            )
        sns.move_legend(ax, "upper right", frameon=True, title=label_input)
        ax.set_title(r"$\textbf{Method:}$" f"{method}")

    def generate_figures(self) -> None:
        methods = ["FG", "MFG", "FGR"]
        scaffold_df, label_encoder = self.get_scaffolds()
        self.plot_scaffolds(label_encoder)

        # Plot alignment
        tsne_fig, tsne_axes = plt.subplots(1, 3, figsize=(12, 3))
        for i, method in enumerate(methods):
            x = self.get_representation(scaffold_df["SMILES"].tolist(), method)
            components = TSNE(random_state=123).fit_transform(x)  # type: ignore
            # Calculate DBI
            dbi = round(
                float(
                    davies_bouldin_score(
                        torch.tensor(x),
                        torch.tensor(scaffold_df["Label"].to_numpy()).reshape(-1),
                    )
                ),
                2,
            )
            self.plot_tsne(
                "Scaffold",
                method,
                dbi,
                components,
                scaffold_df["Label"].to_numpy(),
                ax=tsne_axes[i],
            )
        tsne_fig.savefig(
            f"{self.dataset_dir}/{self.mode}_alignment.png",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
        plt.close()

        # Plot uniformity
        results = {}
        for i, method in enumerate(methods):
            x = self.get_representation(self.smiles, method)
            components = TSNE(random_state=123).fit_transform(x)  # type: ignore
            results[method] = {"x": x, "components": components}

        kde_fig, kde_axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, method in enumerate(methods):
            x = results[method]["x"]
            components = results[method]["components"]
            tsne_norm = components / np.linalg.norm(components, axis=1, keepdims=True)
            self.plot_kde_2d(method, tsne_norm, ax=kde_axes[0, i])
            self.plot_kde(tsne_norm, ax=kde_axes[1, i])
        kde_fig.savefig(
            self.dataset_dir / f"{self.mode}_uniformity.png",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
        plt.close()

        # Plot label alignment
        labels_dir = self.dataset_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Assuming fig.smiles is a list of SMILES strings
        smiles_lengths = [len(smile) for smile in self.smiles]
        smiles_lengths_df = pd.DataFrame(smiles_lengths, columns=["Length"])

        # Calculate the average length
        average_length = smiles_lengths_df["Length"].mean()

        smiles_fig, smiles_axes = plt.subplots(1, 1, figsize=(4, 3))

        # Plot the histogram
        sns.histplot(
            data=smiles_lengths_df,
            x="Length",
            bins=30,
            kde=True,
            color=sns.color_palette(colors)[2],
            ax=smiles_axes,
        )

        # Draw a vertical line at the average length
        smiles_axes.axvline(average_length, color="r", linestyle="--")
        smiles_axes.text(
            average_length + 1,
            smiles_axes.get_ylim()[1] - 10,
            f"Average: {average_length:.2f}",
            color="r",
        )

        smiles_axes.set_title("Length Distribution of SMILES")
        smiles_axes.set_xlabel("Length")
        smiles_axes.set_ylabel("Frequency")
        smiles_fig.savefig(
            labels_dir / f"{self.mode}_smiles_length.png",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
        plt.close()

        for label in self.labels.columns[:30]:
            # Plot label distribution
            label_dist_fig, label_dist_axes = plt.subplots(1, 1, figsize=(4, 3))
            sns.histplot(
                data=self.labels,
                x=label,
                bins=30,
                kde=True,
                color=sns.color_palette(colors)[2],
                ax=label_dist_axes,
            )

            label_dist_axes.set_title("Distribution of Labels")
            label_dist_axes.set_xlabel(label)
            label_dist_axes.set_ylabel("Frequency")
            label_dist_fig.savefig(
                labels_dir / f"{self.mode}_{label}_dist.png",
                bbox_inches="tight",
                dpi=600,
                transparent=True,
            )
            plt.close()

            # Plot label alignment
            label_fig, label_axes = plt.subplots(1, 3, figsize=(12, 3))
            label_values = self.labels[label].to_numpy()
            dbi_scores = [
                round(
                    float(
                        davies_bouldin_score(
                            torch.tensor(results[method]["x"]),
                            torch.tensor(label_values).reshape(-1),
                        )
                    ),
                    2,
                )
                for method in methods
            ]
            for i, method in enumerate(methods):
                self.plot_tsne(
                    label,
                    method,
                    dbi_scores[i],
                    results[method]["components"],
                    label_values,
                    ax=label_axes[i],
                )
            label_fig.savefig(
                labels_dir / f"{self.mode}_{label}_fig.png",
                bbox_inches="tight",
                dpi=600,
                transparent=True,
            )
            plt.close()

    def get_data_model(self, fold_idx: int):
        """Get data and model.

        :param fold_idx: Fold index
        :return: Data and model
        """
        test_smiles = pd.read_parquet(
            self.data_dir
            / "processed"
            / "tasks"
            / self.dataset
            / "splits"
            / "scaffold"
            / f"fold_{fold_idx}"
            / "test.parquet"
        )["SMILES"].to_list()
        x = self.get_representation(test_smiles, "FGR")
        desc = np.stack([get_descriptors(smi) for smi in test_smiles], axis=0)
        ckpt_path = glob.glob(
            f"./models/{self.dataset}/FGR/scaffold/{self.descriptor}/checkpoints/fold_{fold_idx}/epoch_*.ckpt"
        )[0]
        model = FGRLitModule.load_from_checkpoint(ckpt_path).to("cuda:3")
        wrapped_model = WrapperModel(model)
        wrapped_model.eval()
        x = torch.tensor(x, dtype=torch.float32, device=model.device)
        desc = torch.tensor(desc, dtype=torch.float32, device=model.device)
        return x, desc, wrapped_model

    def get_attribution(
        self, attribution_model: Attribution, x: torch.Tensor, desc: torch.Tensor
    ) -> np.ndarray:
        """Get attribution scores from the attribution model.

        :param attribution_model: Attribution model
        :param x: Input tensor
        :param desc: Descriptor tensor
        :return: Attribution scores
        """
        x.requires_grad = True
        desc.requires_grad = True

        # Do not apply baselines for FeaturePermutation
        if isinstance(attribution_model, FeaturePermutation):
            attribution = attribution_model.attribute((x, desc), target=0)
        else:
            baselines = (torch.zeros_like(x), torch.zeros_like(desc))
            attribution = attribution_model.attribute((x, desc), baselines=baselines, target=0)

        attribution_sum = np.concatenate(
            [attr.cpu().detach().numpy() for attr in attribution], axis=1
        ).sum(0)
        norm_attribution = attribution_sum / np.linalg.norm(attribution_sum, ord=2)
        return norm_attribution

    def get_average_attribution(self) -> Dict[str, np.ndarray]:
        """Get average attribution scores.

        :return: Average attribution scores
        """
        methods = {
            "Int Grads": IntegratedGradients,
            "Int Grads w/SmoothGrad": lambda model: NoiseTunnel(IntegratedGradients(model)),
            "GradientShap": GradientShap,
            "Feature Ablation": FeatureAblation,
            "Feature Permutation": FeaturePermutation,
        }

        # Define attributions with the correct type
        attributions: Dict[str, np.ndarray] = {
            name: np.asarray(
                [
                    self.get_attribution(method(model), x, desc)
                    for fold in range(5)
                    for x, desc, model in [self.get_data_model(fold)]
                ]
            )
            for name, method in methods.items()
        }

        # Save the average_attributions dictionary
        with open(self.dataset_dir / "average_attributions.pkl", "wb") as f:
            pickle.dump(attributions, f)

        return attributions

    def plot_attribution(self) -> None:
        """Plot attribution scores."""
        try:
            with open(self.dataset_dir / "average_attributions.pkl", "rb") as f:
                attributions = pickle.load(f)
        except (FileNotFoundError, OSError):
            attributions = self.get_average_attribution()

        average_attributions = {name: np.mean(attr, axis=0) for name, attr in attributions.items()}
        # Calculate the average attribution across methods
        average_attribution = sum(average_attributions.values()) / len(average_attributions)

        # Get the top 20 indices
        final_ind = np.abs(average_attribution).argsort()[-20:]

        feature_names = []
        attribution_values = []
        algorithms = []
        for name, attr in attributions.items():
            for i in range(attr.shape[0]):
                for j in final_ind:
                    algorithms.append(name)
                    feature_names.append(self.input_names[j])
                    attribution_values.append(attr[i, j])
        data = pd.DataFrame(
            {
                "Algorithm": algorithms,
                "Feature": feature_names,
                "Attribution": attribution_values,
            }
        )  # Draw a nested barplot by species and sex
        g = sns.catplot(
            data=data,
            kind="bar",
            y="Attribution",
            x="Feature",
            hue="Algorithm",
            palette=sns.color_palette(colors),
            orient="v",
            height=4,
            aspect=3,
        )

        # Set title
        g.ax.set_title(r"$\textbf{Interpretability Analysis:}$" f"{self.dataset}")

        # Set x and y labels to bold
        g.set_axis_labels("Attribution", "Features")
        g.ax.set_xlabel(g.ax.get_xlabel(), weight="bold")
        g.ax.set_ylabel(g.ax.get_ylabel(), weight="bold")

        g.set_xticklabels(rotation=90)

        # Remove the other legend
        g._legend.remove()  # type: ignore

        # Draw the new legend
        legend = g.ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, 0.95),
            frameon=True,
            title="Interpretability Methods",
        )

        # Set the legend title to bold
        legend.set_title(legend.get_title().get_text(), prop={"weight": "bold"})  # type: ignore
        g.ax.set_xticklabels(
            [
                "\n".join(
                    textwrap.wrap(
                        label.get_text(),
                        width=30,
                    )
                )
                for label in g.ax.get_xticklabels()
            ]
        )
        g.savefig(
            self.dataset_dir / f"{self.mode}_interpretability.png",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
        plt.close()


if __name__ == "__main__":
    df = pd.read_parquet("./data/processed/tasks/summary.parquet")
    tasks = df[df["Datapoints"] < 200000]["Task"].tolist()
    for task in tqdm(tasks):
        for mode in ["input", "train"]:
            try:
                fig_gen = FigureGenerator(dataset=task, mode=mode)
                fig_gen.generate_figures()
                fig_gen.plot_attribution()
            except BaseException:
                print(f"{task} {mode} failed...")
