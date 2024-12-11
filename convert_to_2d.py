"""Script to convert the complete picai public dataset to 2d mid slices.
"""
from picai_eval.image_utils import read_image
from picai_prep.preprocessing import crop_or_pad, resample_img
from cinemri.visualisation import plot_frame
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image

def plot_frame(
    ax,
    image,
    title=None,
    boxes=None,
    box_thickness=3,
    legend=False,
    prediction=None,
    label=None,
    cmap="gray",
    colorbar=False,
):
    """Plot single frame of medical image.

    Parameters
    ----------
    ax : plt.axes object
    image : 2D numpy array
    boxes : list
        List with dictionaries with the following items:
            box : list
                Bounding box in COCO format ([x, y, w, h])
            color : plt color string (optional)
            legend : str (optional)
            label : str (optional)
                Label that will be printed next to the box
            filled : bool (optional)
                if True, this will plot filled boxes with alpha=0.2
    title : str
    legend : bool
        Toggle visibility of legend

    By default this has colormap gray.
    """
    # Plot image
    ax.imshow(image, cmap="gray")

    # Plot bounding box annotations
    if boxes:
        for box in boxes:
            # Process input
            if box["box"] is None:
                continue
            if "color" in box:
                color = box["color"]
            else:
                color = "red"
            if "legend" in box:
                legend_label = box["legend"]
            else:
                legend_label = ""
            if "label" in box:
                box_label = box["label"]
            else:
                box_label = ""
            if "filled" in box:
                if box["filled"]:
                    facecolor = color
                    alpha = 0.4
                else:
                    facecolor = "none"
                    alpha = 1
            else:
                facecolor = "none"
                alpha = 1

            # Zero width/height boxes get a small box
            if box["box"][2] == 0 and box["box"][3] == 0:
                width = 1
                height = 1
            else:
                width = box["box"][2]
                height = box["box"][3]

            rect = patches.Rectangle(
                (box["box"][0], box["box"][1]),
                width,
                height,
                linewidth=box_thickness,
                edgecolor=color,
                facecolor=facecolor,
                label=legend_label,
                alpha=alpha,
            )
            ax.add_patch(rect)
            ax.annotate(box_label, (box["box"][0], box["box"][1] - 2), color=color)

    # Add legend
    if legend:
        add_legend(ax)

    # Add title
    if title is not None:
        ax.set_title(title)

    # Plot prediction heatmap
    imshow = None
    if prediction is not None:
        alpha = 1
        threshold = 0.05
        if cmap is None:
            cmap = get_colormap_from_list("latte", ["lavender", "red"])
        alpha = (prediction > threshold).astype(float) * alpha
        imshow = ax.imshow(prediction, alpha=alpha, cmap=cmap, vmin=0, vmax=1)
        if colorbar:
            plt.colorbar(imshow, fraction=0.046, pad=0.04, ticks=[])

    # Plot ground truth label
    if label is not None:
        enlarged_label = grow_segmentation(label[None, ...].astype(np.float), 2)
        enlarged_label_frame = enlarged_label[0]
        border_only = enlarged_label_frame - label
        alpha = 1
        alpha = (border_only > 0).astype(float) * alpha
        ax.imshow(border_only, alpha=alpha, cmap=cmap)

        # Plot label
        alpha = 0.4
        alpha = (label > 0).astype(float) * alpha
        ax.imshow(label, alpha=alpha, cmap=cmap)

    # Turn off axis
    ax.set_axis_off()

    if imshow:
        return imshow


def plot_middle_slice(t2w, adc, dwi, label, plot_multimodal=True, slice=None):
    n_slices, _, _ = t2w.shape
    if slice is None:
        slice = n_slices // 2

    t2w = t2w[slice]
    adc = adc[slice]
    dwi = dwi[slice]

    # Frankenstein
    multimodal = np.stack([t2w, adc, dwi], axis=2)

    if plot_multimodal:
        return multimodal
    else:
        return t2w

    if False:
        label = label[slice]
        fig, axes = plt.subplots(2, 2)
        axes = axes.ravel()
        for ax, slice in zip(axes, [t2w, adc, dwi]):
            plot_frame(ax, slice, prediction=label)
        axes[-1].imshow(multimodal)
        plt.show()
        quit()

    fig, ax = plt.subplots(figsize=(3, 3))
    if plot_multimodal:
        ax.imshow(multimodal)
        ax.set_axis_off()
    else:
        plot_frame(ax, t2w)

    return fig


def crop_to_fixed_size(image, spacing):
    crop_size = (30, 300, 300)
    reference_spacing = (3, 0.5, 0.5)

    scaling_factors = [reference_spacing[i] / spacing[i] for i in range(3)]
    correct_crop_size = tuple(
        [int(crop_size[i] * scaling_factors[i]) for i in range(3)]
    )

    # Crop
    image = crop_or_pad(image, correct_crop_size)

    return image


def load_single_image(path, is_label=False):
    # Load images
    image = sitk.ReadImage(str(path))

    # Resize to common spacing
    image = resample_img(image, (3, 0.5, 0.5), is_label=is_label)

    # Normalize each individually between 0 and 1
    image = sitk.GetArrayFromImage(image)
    if not is_label:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Crop to prostate region
    image = crop_or_pad(image, (30, 300, 300))

    return image


def is_positive(image_path, labels_path):
    label_path = (
        labels_path
        / "csPCa_lesion_delineations/combined"
        / image_path.name.replace("_t2w.mha", ".nii.gz")
    )
    if label_path.is_file():
        label = load_single_image(label_path, is_label=True)

    if np.max(label) > 0:
        return 1, label
    else:
        return 0, label


def find_mid_slice(image_path, labels_path, positive_status, label):
    # Take median prostate segmentation slice for negative case
    if positive_status == 0:
        seg_path = (
            labels_path
            / "anatomical_delineations/whole_gland/AI/Bosma22b"
            / image_path.name.replace("_t2w.mha", ".nii.gz")
        )
        seg = load_single_image(seg_path, is_label=True)
        prostate_slices = np.unique(np.where(seg == 1)[0])
        mid_slice = int(np.median(prostate_slices))

    # Take slice with highest surface area of cancer for positive case
    if positive_status == 1:
        surface = np.sum(np.sum(label, axis=-1), axis=-1)
        mid_slice = int(np.argmax(surface))

    return mid_slice


def load_image(image_path):
    t2w_path = image_path
    adc_path = image_path.parent / image_path.name.replace("t2w", "adc")
    dwi_path = image_path.parent / image_path.name.replace("t2w", "hbv")

    t2w_image = load_single_image(t2w_path)
    adc_image = load_single_image(adc_path)
    dwi_image = load_single_image(dwi_path)

    return t2w_image, adc_image, dwi_image


if __name__ == "__main__":
    dest_dir = Path(
        "/path/to/dest/dir"
    )
    plot_multimodal = True

    images_path = Path(
        "/path/to/picai/public_training/images"
    )
    labels_path = Path(
        "/path/to/picai_labels"
    )

    for idx, pid in tqdm(enumerate(list(images_path.iterdir()))):
        if not pid.is_dir():
            continue

        for image_path in pid.glob("*t2w.mha"):
            full_id = image_path.name.replace("_t2w.mha", "")

            positive_status, label = is_positive(image_path, labels_path)

            # Find mid slice index
            mid_slice = find_mid_slice(image_path, labels_path, positive_status, label)

            t2w_image, adc_image, dwi_image = load_image(image_path)

            filename = dest_dir / f"{full_id}.png"

            multimodal = plot_middle_slice(
                t2w_image,
                adc_image,
                dwi_image,
                label,
                plot_multimodal=plot_multimodal,
                slice=mid_slice,
            )

            # Save with PIL
            img = Image.fromarray((multimodal * 255).astype(np.uint8))
            img = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
            img.save(filename)
