import time

import numpy as np
import torch
import torchvision.transforms.v2 as v2

from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


class SimpleNestedTorchTransform:
    """
    A picklable wrapper for torchvision / torch transforms.

    This avoids stable_pretraining.transforms.Resize / WrapTorchTransform
    incompatibility with newer torchvision v2 transforms, where Resize may not
    expose the old `.transform` interface.
    """

    def __init__(self, transform, source: str, target: str):
        self.transform = transform
        self.source = source
        self.target = target

    def _split_key(self, key):
        if key is None or key == "":
            return []
        if isinstance(key, (list, tuple)):
            return list(key)
        return str(key).split(".")

    def _get_nested(self, sample, key):
        parts = self._split_key(key)
        cur = sample
        for part in parts:
            cur = cur[part]
        return cur

    def _set_nested(self, sample, key, value):
        parts = self._split_key(key)

        if len(parts) == 0:
            return value

        cur = sample
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]

        cur[parts[-1]] = value
        return sample

    def __call__(self, sample):
        x = self._get_nested(sample, self.source)
        x = self.transform(x)
        return self._set_nested(sample, self.target, x)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """
    Convert image data to tensor/image format, then resize it.

    The original code used:
        dt.transforms.Resize(...)

    That can fail with:
        AttributeError: 'Resize' object has no attribute 'transform'

    So we use torchvision.transforms.v2.Resize directly and wrap it with a
    simple nested-key transform.
    """
    imagenet_stats = dt.dataset_stats.ImageNet

    to_image = dt.transforms.ToImage(
        **imagenet_stats,
        source=source,
        target=target,
    )

    resize_fn = v2.Resize(
        (img_size, img_size),
        antialias=True,
    )

    # After ToImage, read from target and write back to target.
    resize = SimpleNestedTorchTransform(
        resize_fn,
        source=target,
        target=target,
    )

    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """
    Picklable z-score normalizer.

    Uses a class instead of a closure so it survives pickle when DataLoader
    workers are spawned.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]

    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    normalizer = ZScoreNormalizer(mean, std)

    return SimpleNestedTorchTransform(
        normalizer,
        source=source,
        target=target,
    )


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f"weights_epoch_{epoch}.pt",
        )


class JEPATrainStatsCallback(Callback):
    """Log optimizer and throughput stats without changing training behavior."""

    def __init__(self):
        super().__init__()
        self._batch_start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_start_time is None:
            return

        elapsed = max(time.perf_counter() - self._batch_start_time, 1e-12)
        batch_size = self._get_batch_size(batch)
        if batch_size is not None:
            world_size = getattr(trainer, "world_size", 1) or 1
            samples_per_sec = float(batch_size * world_size) / elapsed
            pl_module.log(
                "fit/samples_per_sec",
                samples_per_sec,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer is not None:
            for group_idx, group in enumerate(optimizer.param_groups):
                pl_module.log(
                    f"fit/learning_rate_{group_idx}",
                    group.get("lr", 0.0),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )

    def _get_batch_size(self, batch):
        if isinstance(batch, dict):
            for value in batch.values():
                size = self._get_batch_size(value)
                if size is not None:
                    return size
            return None

        if isinstance(batch, (list, tuple)):
            for value in batch:
                size = self._get_batch_size(value)
                if size is not None:
                    return size
            return None

        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.size(0))

        return None
