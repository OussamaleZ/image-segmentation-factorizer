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
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
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
        return datalist

    def _split_datalist(self, datalist: List[dict]) -> Tuple[List[dict], List[dict]]:
        rng = random.Random(self.seed)
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
        """
        Training transforms:
        - Load image & label
        - Channel-first for both
        - Intensity scaling + normalization for image
        - Label-aware cropping around tumor classes (nnU-Net / Factorizer style)
        - Spatial augmentations
        """
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),

                # Intensity preprocessing for images
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.0,
                    a_max=5000.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

                # 🔥 Label-aware cropping: focus on tumor classes instead of random crops
                # Classes: 0=background, 1,2,3=tumor subregions (3=ET)
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.roi_size,
                    num_classes=4,
                    num_samples=1,
                    # ratios: relative frequency for sampling patches containing each class
                    # Here we ignore background (0) and encourage tumor classes.
                    # You can tune these (e.g. [0, 1, 1, 2] to oversample ET even more).
                    ratios=[0.0, 1.0, 1.0, 2.0],
                ),

                # Data augmentation
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def _build_val_transforms(self) -> Compose:
        """
        Validation transforms:
        - No cropping: work on full volumes for sliding window inference.
        """
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

        # Train dataset
        train_kwargs = {"data": train_files, "transform": self.train_transforms}
        if self.use_cache:
            train_kwargs["cache_rate"] = self.cache_rate
        train_ds = dataset_cls(**train_kwargs)

        # Val dataset
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

    # ------------------------------------------------------------------
    # Post transforms (optional)
    # ------------------------------------------------------------------
    def post_transforms(self) -> Compose:
        return Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True),
            ]
        )
