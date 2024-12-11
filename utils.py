import os
import pandas as pd
from sklearn.metrics import auc, roc_curve
from picai_baseline.splits.picai import train_splits, valid_splits
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union
import pickle


def load_marksheet(picai_labels_path):
    df = pd.read_csv(picai_labels_path / "clinical_information" / "marksheet.csv")
    # Add column with binary reference used in challenge
    df["binary"] = df["lesion_GS"].apply(binarize_lesion_GS)

    # Impute the psad values
    df = impute_psad(df)

    return df


def impute_psad(marksheet):
    for idx, row in marksheet.iterrows():
        if np.isnan(row["psad"]):
            if not np.isnan(row["psa"]) and not np.isnan(row["prostate_volume"]):
                marksheet.loc[idx, "psad"] = row["psa"] / row["prostate_volume"]

    return marksheet


def load_train_marksheet(fold):
    marksheet = load_marksheet()
    train_split = train_splits[fold]["subject_list"]
    study_ids = [int(s.split("_")[1]) for s in train_split]

    binary_mask = [study_id in study_ids for study_id in marksheet["study_id"]]
    marksheet_train = marksheet[binary_mask]

    return marksheet_train


def load_val_marksheet(fold):
    marksheet = load_marksheet()
    val_split = valid_splits[fold]["subject_list"]
    study_ids = [int(s.split("_")[1]) for s in val_split]

    binary_mask = [study_id in study_ids for study_id in marksheet["study_id"]]
    marksheet_val = marksheet[binary_mask]

    return marksheet_val


def binarize_lesion_GS(finding_raw):
    finding = str(finding_raw)
    # Positives
    if "3+4" in finding:
        return 1
    elif "4+3" in finding:
        return 1
    elif "4+4" in finding:
        return 1
    elif "4+5" in finding:
        return 1
    elif "5+4" in finding:
        return 1
    elif "5+3" in finding:
        return 1
    elif "3+5" in finding:
        return 1
    elif "5+5" in finding:
        return 1
    # Negatives
    elif finding == "nan":
        return 0
    else:
        return 0


def calculate_ROC(y_det, y_true):
    """
    Generate Receiver Operating Characteristic curve for case-level risk stratification.
    """
    fpr, tpr, thresholds = roc_curve(
        y_true=y_true,
        y_score=y_det,
    )
    auroc = auc(fpr, tpr)

    return fpr, tpr, auroc, thresholds


def plot_frankenstein(image):
    t2w = image[:, :, 0]
    adc = image[:, :, 1]
    dwi = image[:, :, 2]

    n_cols = 4

    fig, axes = plt.subplots(1, n_cols)
    axes = axes.ravel()
    for ax, slice, label in zip(axes, [t2w, adc, dwi], ["t2w", "adc", "dwi"]):
        ax.imshow(slice, cmap="gray")
        ax.set_title(label)

    if plot_frankenstein:
        axes[-1].imshow(image)
        axes[-1].set_title("frankenstein")

    return fig


def _flatten_dict(params, delimiter: str = "/", parent_key: str = ""):
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """
    result = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if isinstance(v, Namespace):
            v = vars(v)
        if isinstance(v, MutableMapping):
            result = {
                **result,
                **_flatten_dict(v, parent_key=new_key, delimiter=delimiter),
            }
        else:
            result[new_key] = v
    return result


def load_public_to_pubpriv():
    with open("public_to_pubpriv.pkl", "rb") as pkl_file:
        return pickle.load(pkl_file)
