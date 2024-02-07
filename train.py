import hydra
import torchmetrics
import pytorch_lightning as pl
import wandb
import gc
from utils import _flatten_dict

from flash.image import ImageClassificationData, ImageClassifier
from pathlib import Path
import os

from augmentation import transform_objects


def get_files_and_targets(image_dir):
    files = []
    targets = []
    for filepath in (image_dir / "neg").glob("*.png"):
        files.append(filepath)
        targets.append(0)
    for filepath in (image_dir / "pos").glob("*.png"):
        files.append(filepath)
        targets.append(1)

    return files, targets


def train_loop(cfg):
    data_dir = Path(cfg.data_dir)

    train_files, train_targets = get_files_and_targets(data_dir / "train")
    val_files, val_targets = get_files_and_targets(data_dir / "val")
    test_files, test_targets = get_files_and_targets(data_dir / "test")

    datamodule = ImageClassificationData.from_files(
        train_files=train_files,
        train_targets=train_targets,
        val_files=val_files,
        val_targets=val_targets,
        test_files=test_files,
        test_targets=test_targets,
        transform=transform_objects[cfg.train.aug_level],
        **cfg.datamodule
    )

    # 2. Build the task
    model = ImageClassifier(
        num_classes=2,
        backbone=cfg.architecture,
        labels=datamodule.labels,
        pretrained=True,
        learning_rate=cfg.train.lr,
        metrics=[
            torchmetrics.classification.MulticlassAUROC(num_classes=2),
            torchmetrics.Accuracy(),
        ],
    )

    # 3. Create the trainer and finetune the model
    loggers = []
    if cfg.log.wandb:
        wandb_logger = pl.loggers.WandbLogger(config=_flatten_dict(cfg), **cfg.wandb)
        loggers.append(wandb_logger)
    callbacks = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_multiclassauroc", mode="max", save_last=False
    )
    callbacks.append(checkpoint_callback)
    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model, datamodule=datamodule)

    # 4. Test on test set
    trainer.test(ckpt_path="best", dataloaders=datamodule)

    # Log parameters to visualize as group in wandb
    if cfg.log.wandb:
        wandb.finish()
        del wandb_logger

    # Clear RAM
    del trainer
    del model
    del datamodule
    gc.collect()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # Run training loop
    for i in range(cfg.train.n_runs):
        train_loop(cfg)


if __name__ == "__main__":
    main()
