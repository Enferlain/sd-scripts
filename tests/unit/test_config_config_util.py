import types
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch, MagicMock

from library.config.config_util import (
    BlueprintGenerator,
    DatasetBlueprint,
    DatasetGroupBlueprint,
    generate_dataset_group_by_blueprint,
    generate_dreambooth_subsets_config_by_subdirs,
    generate_user_config_from_dataset,
)

# ============================================================================
# Mocks & Fakes
# ============================================================================

@dataclass
class FakeSubsetCfg:
    image_dir: str = "train"
    num_repeats: int = 1
    metadata_file: str | None = None
    conditioning_data_dir: str | None = None
    # Add other fields accessed by BlueprintGenerator.generate loop
    shuffle_caption: bool = False
    keep_tokens: int = 0
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0 
    caption_tag_dropout_rate: float = 0.0
    caption_prefix: str | None = None
    caption_suffix: str | None = None
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: str | None = None
    random_crop: bool = False
    token_warmup_min: int = 1
    token_warmup_step: float = 0
    alpha_mask: bool = False
    resize_interpolation: str | None = None
    custom_attributes: str | None = None

@dataclass
class FakeDatasetConfig:
    subsets: list
    # dataset-level fields
    resolution: list[int] | None = None
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    # for dreambooth user_config
    dataset_class: str | None = None
    train_data_dir: str | None = None
    reg_data_dir: str | None = None
    # other fields accessed
    adapter_multiplier: float = 1.0
    debug_dataset: bool = False
    validation_seed: int | None = None
    validation_split: float = 0.0
    resize_interpolation: str | None = None
    prior_loss_weight: float = 1.0

@dataclass
class FakeBucketsConfig:
    resolution: list[int] | None = None
    min_bucket_reso: int | None = None
    max_bucket_reso: int | None = None
    bucket_reso_steps: int | None = None
    bucket_no_upscale: bool | None = None

def make_root_cfg(dataset_cfg, buckets_cfg=None):
    root = types.SimpleNamespace()
    root.dataset = dataset_cfg
    root.buckets = buckets_cfg or FakeBucketsConfig()
    return root

# ============================================================================
# BlueprintGenerator Tests
# ============================================================================

def test_blueprint_dreambooth_type():
    subset = FakeSubsetCfg(image_dir="db", num_repeats=2)
    ds_cfg = FakeDatasetConfig(subsets=[subset], resolution=[512, 512])
    root = make_root_cfg(ds_cfg)

    bg = BlueprintGenerator()
    bp = bg.generate(root)

    assert len(bp.dataset_group.datasets) == 1
    db = bp.dataset_group.datasets[0]
    assert db.is_dreambooth is True
    assert db.is_controlnet is False
    # resolution list -> tuple
    assert db.params.resolution == (512, 512)
    # subset params merged
    sp = db.subsets[0].params
    assert sp.image_dir == "db"
    assert sp.num_repeats == 2

def test_blueprint_finetune_type():
    subset = FakeSubsetCfg(image_dir="ft", num_repeats=1, metadata_file="meta.json")
    ds_cfg = FakeDatasetConfig(subsets=[subset])
    root = make_root_cfg(ds_cfg)

    bg = BlueprintGenerator()
    bp = bg.generate(root)

    db = bp.dataset_group.datasets[0]
    assert db.is_dreambooth is False
    assert db.is_controlnet is False
    # Check param type by checking available fields or instance type name if accessible
    assert type(db.params).__name__ == "FineTuningDatasetParams"
    assert db.subsets[0].params.metadata_file == "meta.json"

@dataclass
class FakeControlNetSubsetCfg:
    image_dir: str
    conditioning_data_dir: str
    num_repeats: int = 1
    
def test_blueprint_controlnet_type():
    subset = FakeControlNetSubsetCfg(image_dir="cn", conditioning_data_dir="conds")
    ds_cfg = FakeDatasetConfig(subsets=[subset])
    root = make_root_cfg(ds_cfg)

    bg = BlueprintGenerator()
    bp = bg.generate(root)

    db = bp.dataset_group.datasets[0]
    assert db.is_dreambooth is False
    assert db.is_controlnet is True
    assert type(db.params).__name__ == "ControlNetDatasetParams"
    assert db.subsets[0].params.conditioning_data_dir == "conds"

def test_blueprint_uses_buckets_defaults():
    subset = FakeSubsetCfg(image_dir="db")
    ds_cfg = FakeDatasetConfig(subsets=[subset], resolution=None)
    buckets = FakeBucketsConfig(resolution=[256, 384])
    root = make_root_cfg(ds_cfg, buckets_cfg=buckets)
    
    bg = BlueprintGenerator()
    bp = bg.generate(root)
    db = bp.dataset_group.datasets[0]
    
    # Needs to take resolution from buckets config if dataset config is None
    assert db.params.resolution == (256, 384)

# ============================================================================
# generate_dataset_group_by_blueprint Tests
# ============================================================================

class FakeDatasetBase:
    def __init__(self, subsets=None, **kwargs):
        self.subsets = subsets or []
        self.kwargs = kwargs
        # Set default attributes accessed by print_info
        self.batch_size = kwargs.get("batch_size", 1)
        self.enable_bucket = kwargs.get("enable_bucket", False)
        self.min_bucket_reso = 256
        self.max_bucket_reso = 1024
        self.bucket_reso_steps = 64
        self.bucket_no_upscale = False
        res = kwargs.get("resolution", (512, 512))
        self.width, self.height = res if res else (512, 512)
        self.resize_interpolation = "lanczos"
        self.image_data = {}
        self.image_to_subset = {}
        self.num_train_images = 0
        self.num_reg_images = 0
        
    def make_buckets(self):
        pass
    
    def set_seed(self, seed):
        pass

    def __len__(self):
        return 1

class FakeDreamBoothDataset(FakeDatasetBase):
    pass

class FakeFineTuningDataset(FakeDatasetBase):
    pass

class FakeControlNetDataset(FakeDatasetBase):
    pass

class FakeSubsetBase:
    def __init__(self, **kwargs):
        self.params = kwargs
        # Attributes accessed by print_info
        self.image_dir = kwargs.get("image_dir", "dir")
        self.img_count = 10
        self.num_repeats = kwargs.get("num_repeats", 1)
        self.shuffle_caption = False
        self.keep_tokens = 0
        self.caption_dropout_rate = 0.0
        self.caption_dropout_every_n_epochs = 0
        self.caption_tag_dropout_rate = 0.0
        self.caption_prefix = None
        self.caption_suffix = None
        self.color_aug = False
        self.flip_aug = False
        self.face_crop_aug_range = None
        self.random_crop = False
        self.token_warmup_min = 1
        self.token_warmup_step = 0
        self.alpha_mask = False
        self.resize_interpolation = None
        self.custom_attributes = None
        # Specifics
        self.is_reg = kwargs.get("is_reg", False)
        self.class_tokens = kwargs.get("class_tokens", "class")
        self.caption_extension = ".txt"
        self.metadata_file = kwargs.get("metadata_file", None)

class FakeDreamBoothSubset(FakeSubsetBase):
    pass
    
class FakeFineTuningSubset(FakeSubsetBase):
    pass

class FakeControlNetSubset(FakeSubsetBase):
    pass

@patch("library.config.config_util.DreamBoothDataset", new=FakeDreamBoothDataset)
@patch("library.config.config_util.FineTuningDataset", new=FakeFineTuningDataset)
@patch("library.config.config_util.ControlNetDataset", new=FakeControlNetDataset)
@patch("library.config.config_util.DreamBoothSubset", new=FakeDreamBoothSubset)
@patch("library.config.config_util.FineTuningSubset", new=FakeFineTuningSubset)
@patch("library.config.config_util.ControlNetSubset", new=FakeControlNetSubset)
def test_generate_dataset_group_dreambooth():
    from library.config.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams, SubsetBlueprint

    subset_params = DreamBoothSubsetParams(image_dir="db", num_repeats=2)
    ds_params = DreamBoothDatasetParams(batch_size=3, resolution=(512, 512))
    bp = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=ds_params,
        subsets=[SubsetBlueprint(params=subset_params)],
    )
    group_bp = DatasetGroupBlueprint(datasets=[bp])

    dg, val_dg = generate_dataset_group_by_blueprint(group_bp)

    assert dg is not None
    assert val_dg is None
    
    # Verify dataset type and content
    ds = dg.datasets[0]
    assert isinstance(ds, FakeDreamBoothDataset)
    assert ds.batch_size == 3
    assert ds.kwargs.get("is_training_dataset") is True
    
    # Verify subset
    assert len(ds.subsets) == 1
    sub = ds.subsets[0]
    assert isinstance(sub, FakeDreamBoothSubset)
    assert sub.num_repeats == 2

@patch("library.config.config_util.DreamBoothDataset", new=FakeDreamBoothDataset)
@patch("library.config.config_util.DreamBoothSubset", new=FakeDreamBoothSubset)
def test_generate_dataset_group_validation_split():
    from library.config.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams, SubsetBlueprint

    subset_params = DreamBoothSubsetParams(image_dir="db")
    ds_params = DreamBoothDatasetParams(validation_split=0.5)
    bp = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=ds_params,
        subsets=[SubsetBlueprint(params=subset_params)],
    )
    group_bp = DatasetGroupBlueprint(datasets=[bp])

    dg, val_dg = generate_dataset_group_by_blueprint(group_bp)

    assert dg is not None
    assert val_dg is not None
    
    # Train dataset
    train_ds = dg.datasets[0]
    assert isinstance(train_ds, FakeDreamBoothDataset)
    assert train_ds.kwargs.get("is_training_dataset") is True
    
    # Val dataset
    val_ds = val_dg.datasets[0]
    assert isinstance(val_ds, FakeDreamBoothDataset)
    assert val_ds.kwargs.get("is_training_dataset") is False

@patch("library.config.config_util.DreamBoothDataset", new=FakeDreamBoothDataset)
@patch("library.config.config_util.DreamBoothSubset", new=FakeDreamBoothSubset)
def test_generate_dataset_group_validation_split_zero():
    """When validation_split=0.0, no validation dataset should be created."""
    from library.config.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams, SubsetBlueprint

    subset_params = DreamBoothSubsetParams(image_dir="db")
    ds_params = DreamBoothDatasetParams(validation_split=0.0)
    bp = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=ds_params,
        subsets=[SubsetBlueprint(params=subset_params)],
    )
    group_bp = DatasetGroupBlueprint(datasets=[bp])

    dg, val_dg = generate_dataset_group_by_blueprint(group_bp)

    assert dg is not None
    assert val_dg is None  # No validation dataset when split is 0.0

@patch("library.config.config_util.DreamBoothDataset", new=FakeDreamBoothDataset)
@patch("library.config.config_util.DreamBoothSubset", new=FakeDreamBoothSubset)
def test_generate_dataset_group_validation_split_invalid(caplog):
    """When validation_split > 1.0, a warning should be logged and no val dataset created."""
    from library.config.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams, SubsetBlueprint
    import logging

    subset_params = DreamBoothSubsetParams(image_dir="db")
    ds_params = DreamBoothDatasetParams(validation_split=1.5)  # Invalid: > 1.0
    bp = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=ds_params,
        subsets=[SubsetBlueprint(params=subset_params)],
    )
    group_bp = DatasetGroupBlueprint(datasets=[bp])

    with caplog.at_level(logging.WARNING):
        dg, val_dg = generate_dataset_group_by_blueprint(group_bp)

    assert dg is not None
    assert val_dg is None  # Invalid split should skip validation dataset
    assert "not a valid number" in caplog.text

# Track seed calls for verification
class SeedTrackingDataset(FakeDatasetBase):
    seeds_received = []
    
    def set_seed(self, seed):
        SeedTrackingDataset.seeds_received.append(seed)

@patch("library.config.config_util.random.randint", return_value=42)
@patch("library.config.config_util.DreamBoothDataset", new=SeedTrackingDataset)
@patch("library.config.config_util.DreamBoothSubset", new=FakeDreamBoothSubset)
def test_dataset_group_seed_consistency(mock_randint):
    """All datasets should receive the same random seed for reproducibility."""
    from library.config.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams, SubsetBlueprint
    
    SeedTrackingDataset.seeds_received = []  # Reset tracking
    
    bp1 = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=DreamBoothDatasetParams(),
        subsets=[SubsetBlueprint(params=DreamBoothSubsetParams(image_dir="ds1"))],
    )
    bp2 = DatasetBlueprint(
        is_dreambooth=True,
        is_controlnet=False,
        params=DreamBoothDatasetParams(),
        subsets=[SubsetBlueprint(params=DreamBoothSubsetParams(image_dir="ds2"))],
    )
    group_bp = DatasetGroupBlueprint(datasets=[bp1, bp2])
    
    dg, _ = generate_dataset_group_by_blueprint(group_bp)
    
    # random.randint should be called exactly once
    mock_randint.assert_called_once()
    
    # Both datasets should have received the same seed (42)
    assert len(SeedTrackingDataset.seeds_received) == 2
    assert SeedTrackingDataset.seeds_received[0] == 42
    assert SeedTrackingDataset.seeds_received[1] == 42

# ============================================================================
# generate_dreambooth_subsets_config_by_subdirs Tests
# ============================================================================

def test_generate_dreambooth_subdirs_basic(tmp_path):
    (tmp_path / "10_cat").mkdir()
    (tmp_path / "5_dog").mkdir()
    (tmp_path / "invalid").mkdir()

    configs = generate_dreambooth_subsets_config_by_subdirs(
        train_data_dir=str(tmp_path),
        reg_data_dir=None,
    )

    assert len(configs) == 2
    reps = {c["image_dir"]: c["num_repeats"] for c in configs}
    classes = {c["image_dir"]: c["class_tokens"] for c in configs}
    
    # Verify parsing
    # Note: image_dir path will be absolute string
    assert any(str(tmp_path / "10_cat") in k and v == 10 for k, v in reps.items())
    assert any(str(tmp_path / "5_dog") in k and v == 5 for k, v in reps.items())
    assert any("cat" in v for v in classes.values())
    assert all(c["is_reg"] is False for c in configs)

def test_generate_dreambooth_subdirs_with_reg(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "1_person").mkdir()
    
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir()
    (reg_dir / "1_man").mkdir()
    
    configs = generate_dreambooth_subsets_config_by_subdirs(
        train_data_dir=str(train_dir),
        reg_data_dir=str(reg_dir)
    )
    
    assert len(configs) == 2
    reg_configs = [c for c in configs if c["is_reg"]]
    train_configs = [c for c in configs if not c["is_reg"]]
    
    assert len(reg_configs) == 1
    assert "man" in reg_configs[0]["class_tokens"]
    
    assert len(train_configs) == 1
    assert "person" in train_configs[0]["class_tokens"]

# ============================================================================
# generate_user_config_from_dataset Tests
# ============================================================================

@patch("library.config.config_util.generate_dreambooth_subsets_config_by_subdirs")
def test_generate_user_config_from_dataset_dreambooth(mock_gen):
    mock_gen.return_value = [{"image_dir": "x", "num_repeats": 1}]
    ds_cfg = FakeDatasetConfig(
        subsets=[],
        dataset_class=None,
        train_data_dir="train_dir",
        reg_data_dir="reg_dir",
    )
    user_cfg = generate_user_config_from_dataset(ds_cfg)

    mock_gen.assert_called_once()
    assert "datasets" in user_cfg
    assert user_cfg["datasets"][0]["subsets"] == mock_gen.return_value

def test_generate_user_config_from_dataset_arbitrary():
    ds_cfg = FakeDatasetConfig(
        subsets=[],
        dataset_class="some.class",
    )
    user_cfg = generate_user_config_from_dataset(ds_cfg)
    
    # Arbitrary dataset class usage returns empty datasets list structure
    # logic: if dataset_config.dataset_class is None: ... else: user_config = {"datasets": []}
    assert user_cfg == {"datasets": []}
