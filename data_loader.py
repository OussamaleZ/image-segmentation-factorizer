import math
import random
from pathlib import Path
from typing import List, Tuple

from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist, list_data_collate
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class data_laoder:
    """Prepare MONAI datasets and dataloaders for 3D medical image segmentation."""

    def __init__(
        self,
        data_dir: str = "data/brats",
        dataset_json: str = "dataset.json",
        batch_size: int = 1,
        val_frac: float = 0.2,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        num_workers: int = 4,
        cache_rate: float = 0.1,
        seed: int = 42,
        use_cache: bool = True,
        data_frac: float = 1.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_json = self.data_dir / dataset_json
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.roi_size = roi_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.seed = seed
        self.use_cache = use_cache
        self.data_frac = data_frac

        if not self.dataset_json.exists():
            raise FileNotFoundError(f"Expected dataset json at {self.dataset_json}")

        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()

    # ------------------------------------------------------------------
    # Dataset list and split
    # ------------------------------------------------------------------
    def _load_datalist(self) -> List[dict]:
        datalist = load_decathlon_datalist(
            str(self.dataset_json),
            data_list_key="training",
            base_dir=str(self.data_dir),
        )
        if len(datalist) == 0:
            raise RuntimeError(f"No training entries found in {self.dataset_json}")

        orig_len = len(datalist)
        rng = random.Random(self.seed)
        rng.shuffle(datalist)

        if 0.0 < self.data_frac < 1.0:
            n_keep = max(1, int(orig_len * self.data_frac))
            datalist = datalist[:n_keep]
            print(
                f"[data_laoder] Using only {n_keep}/{orig_len} cases "
                f"(~{100.0 * self.data_frac:.1f} % of the data)"
            )

        return datalist
    def _build_custom_loader(self, list_of_files, is_train=True):
        dataset_cls = CacheDataset if self.use_cache else Dataset

        transforms = self.train_transforms if is_train else self.val_transforms

        kwargs = {"data": list_of_files, "transform": transforms}
        if self.use_cache:
            kwargs["cache_rate"] = self.cache_rate
        ds = dataset_cls(**kwargs)

        return DataLoader(
            ds,
            batch_size=self.batch_size if is_train else 1,
            shuffle=is_train,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
        )

    def _split_datalist(self, datalist: List[dict]) -> Tuple[List[dict], List[dict]]:
        rng = random.Random(self.seed + 1)
        rng.shuffle(datalist)

        split_idx = int(math.ceil(len(datalist) * (1 - self.val_frac)))
        train_files = datalist[:split_idx]
        val_files = datalist[split_idx:]
        if len(val_files) == 0:
            val_files = train_files[-1:]
            train_files = train_files[:-1]
        return train_files, val_files

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def _build_train_transforms(self) -> Compose:
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),

                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.0,
                    a_max=5000.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.roi_size,
                    num_classes=4,
                    num_samples=1,
                    ratios=[0.0, 1.0, 1.0, 2.0],
                ),

                # Geometric aug
                RandAffined(
                    keys=["image", "label"],
                    prob=0.15,
                    rotate_range=(0.5235988, 0.5235988, 0.5235988),  # ±30°
                    scale_range=(0.3, 0.3, 0.3),  # 1±0.3 → [0.7, 1.3]
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

                # Intensity aug
                RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.15,
                    sigma_x=(0.5, 1.5),
                    sigma_y=(0.5, 1.5),
                    sigma_z=(0.5, 1.5),
                ),
                RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.3),
                RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),

                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def _build_val_transforms(self) -> Compose:
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.0,
                    a_max=5000.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        datalist = self._load_datalist()
        train_files, val_files = self._split_datalist(datalist)

        dataset_cls = CacheDataset if self.use_cache else Dataset

        train_kwargs = {"data": train_files, "transform": self.train_transforms}
        if self.use_cache:
            train_kwargs["cache_rate"] = self.cache_rate
        train_ds = dataset_cls(**train_kwargs)

        val_kwargs = {"data": val_files, "transform": self.val_transforms}
        if self.use_cache:
            val_kwargs["cache_rate"] = self.cache_rate
        val_ds = dataset_cls(**val_kwargs)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
        )
        return train_loader, val_loader

    def post_transforms(self) -> Compose:
        return Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True),
            ]
        )
