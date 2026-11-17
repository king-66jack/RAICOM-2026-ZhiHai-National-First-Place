# -*- coding: utf-8 -*-


import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import warnings
from contextlib import nullcontext
import csv
import json
import random
import shutil
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets

import timm
from timm.utils import ModelEmaV2
from safetensors.torch import load_file as load_safetensors

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_warning_filters():
    """只屏蔽当前环境中已知的非致命兼容/性能提示，不屏蔽真实报错。"""
    warnings.filterwarnings(
        "ignore",
        message="A new version of Albumentations is available.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Grad strides do not match bucket view strides.*",
        category=UserWarning,
    )


def amp_autocast(device: torch.device, enabled: bool):
    """AMP autocast 兼容封装。

    AMP autocast 兼容封装，不要求升级 PyTorch：
    - 新版优先使用 torch.amp.autocast
    - 旧版自动回退 torch.cuda.amp.autocast
    - CPU 或 disabled 时返回 nullcontext
    """
    enabled = bool(enabled) and device.type == "cuda"
    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except TypeError:
            try:
                return torch.amp.autocast("cuda", enabled=True)
            except TypeError:
                pass

    return torch.cuda.amp.autocast(enabled=True)


def make_amp_scaler(enabled: bool):
    """GradScaler 兼容封装，不要求升级 PyTorch。"""
    enabled = bool(enabled)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=enabled)
            except TypeError:
                pass

    return torch.cuda.amp.GradScaler(enabled=enabled)


# =========================================================
# 1. 大参数表：公共训练、增强、评估参数统一在这里修改
# =========================================================
PARAMS: Dict[str, Any] = {
    # 默认实验，可通过命令行 --experiment 覆盖。
    "active_experiment": "E0_convnextv2_tiny",

    # ImageFolder 数据目录：train/<class_name>/xxx.jpg，val/<class_name>/xxx.jpg
    "train_dir": "",
    "val_dir": "",

    # 类别由 ImageFolder 子目录自动推断。
    "num_classes": 0,
    "dataset_classes": [],
    "class_alias_for_output": {},

    # 输出目录。
    "root_save_path": "",

    # ------------------------- 公共训练参数 -------------------------
    "epochs": 50,
    "batch_size": 32,          # 各模型可以在 EXPERIMENTS 中覆盖
    "num_workers": 8,
    "lr": 3e-4,               # 各模型可以在 EXPERIMENTS 中覆盖
    "weight_decay": 1.5e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-7,
    "patience": 10,
    "grad_clip": 1.0,
    "label_smoothing": 0.03,
    "seed": 2024,

    # 训练稳定性。
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.9992,
    "use_sync_bn": True,
    "channels_last": False,
    "suppress_known_warnings": True,

    # 默认输入尺寸；各模型会在 EXPERIMENTS 中覆盖。
    "img_size": 224,

    # 使用 timm 模型自带的预训练归一化参数。
    # build_model 后会通过 timm.data.resolve_model_data_config 自动更新这两项。
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],

    # ------------------------- 增强总开关 -------------------------
    # 三大类增强独立触发；每一大类内部最多随机选择一种。
    "use_physical_degradation": True,
    "use_photometric_aug": True,
    "use_occlusion_aug": True,
    "aug_physical_group_p": 0.45,
    "aug_photometric_group_p": 0.35,
    "aug_occlusion_group_p": 0.18,

    # ------------------- 第一类：物理退化与天气模拟 -------------------
    # 雨。
    "aug_rain_slant": [-10, 10],
    "aug_rain_drop_length": 10,
    "aug_rain_drop_width": 1,
    "aug_rain_blur_value": 3,
    "aug_rain_brightness": 0.90,

    # 雪。
    "aug_snow_point": [0.08, 0.22],
    "aug_snow_brightness": 1.8,

    # 雾。
    "aug_fog_coef": [0.10, 0.35],
    "aug_fog_alpha": 0.08,

    # 沙尘：自定义暖色雾化层，强度不宜太高，避免完全改变标识颜色。
    "aug_sand_alpha": [0.08, 0.22],
    "aug_sand_blur_limit": [3, 7],
    "aug_sand_noise_std": [2.0, 10.0],

    # 光学与传感器退化。
    "aug_motion_blur_limit": [3, 9],
    "aug_gaussian_blur_limit": [3, 7],
    "aug_iso_color_shift": [0.01, 0.05],
    "aug_iso_intensity": [0.10, 0.45],
    "aug_sunflare_roi": [0.0, 0.0, 1.0, 1.0],
    "aug_sunflare_radius_ratio": 0.18,

    # JPEG 压缩。10~30 很激进，默认稍微放宽到 15~45；可在这里直接改成 [10, 30]。
    "aug_jpeg_quality": [15, 45],

    # ------------------- 第二类：颜色与光度变换 -------------------
    "aug_hue_shift_limit": 10,
    "aug_sat_shift_limit": 25,
    "aug_val_shift_limit": 25,
    "aug_brightness_limit": [-0.40, 0.15],
    "aug_contrast_limit": [-0.25, 0.30],
    "aug_clahe_clip_limit": [2.0, 4.0],
    "aug_channel_dropout_fill": 0,
    # ToGray 和 ChannelDropout 都放在 OneOf 中，实际触发率低于大类总概率。

    # ------------------- 第三类：局部遮挡与破坏 -------------------
    "aug_coarse_holes": [1, 3],
    "aug_coarse_hole_ratio": [0.06, 0.16],
    "aug_grid_ratio": 0.35,
    "aug_grid_unit_ratio": [0.12, 0.24],
    "aug_dropout_fill": 0,

    # ------------------------- 明确禁用 -------------------------
    # 这些增强可能改变“左转/右转、靠左/靠右”等类别语义，或裁掉主体。
    "use_horizontal_flip": False,
    "use_vertical_flip": False,
    "use_rotation": False,
    "use_random_crop": False,
    "use_mixup": False,
    "use_cutmix": False,
    "use_randaugment": False,

    # ------------------------- 评估输出 -------------------------
    "save_eval_outputs": True,
    "copy_wrong_images": False,
    "save_probs_in_csv": True,

    # 本地预训练权重保护：匹配比例太低时直接报错，避免实际随机初始化。
    "min_pretrained_match_ratio": 0.50,
}


# =========================================================
# 2. 四模型实验表：只放模型相关差异，增强策略全部共用 PARAMS
# =========================================================
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "E0_convnextv2_tiny": {
        "stage": "E0",
        "role": "ensemble_member_1",
        "model_tag": "traffic_convnextv2_tiny",
        "model_arch": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "model_family": "convnextv2",
        "pretrained": False,
        "require_pretrained": True,
        "local_weight_path": "",
        "drop_path_rate": 0.15,
        "batch_size": 32,
        "lr": 2.5e-4,
        "img_size": 224,
    },

    "E1_eva02_base": {
        "stage": "E1",
        "role": "ensemble_member_2",
        "model_tag": "traffic_eva02_base",
        "model_arch": "eva02_base_patch14_224.mim_in22k",
        "model_family": "eva02",
        "pretrained": False,
        "require_pretrained": True,
        "local_weight_path": "",
        "drop_path_rate": 0.10,
        "batch_size": 16,
        "lr": 5e-5,
        "img_size": 224,
    },

    "E2_maxxvitv2_base": {
        "stage": "E2",
        "role": "ensemble_member_3",
        "model_tag": "traffic_maxxvitv2_base",
        "model_arch": "maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        "model_family": "maxxvitv2",
        "pretrained": False,
        "require_pretrained": True,
        "local_weight_path": "",
        "drop_path_rate": 0.12,
        "batch_size": 8,
        "lr": 5e-5,
        "img_size": 224,
    },

    "E3_swinv2_base": {
        "stage": "E3",
        "role": "ensemble_member_4",
        "model_tag": "traffic_swinv2_base",
        "model_arch": "swinv2_base_window12_192.ms_in22k",
        "model_family": "swinv2",
        "pretrained": False,
        "require_pretrained": True,
        "local_weight_path": "",
        "drop_path_rate": 0.15,
        "batch_size": 16,
        "lr": 8e-5,
        "img_size": 192,
    }
}


# =========================================================
# 3. 基础工具函数
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交通标识四模型统一训练脚本")
    parser.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--root_save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--local_weight_path", type=str, default=None)
    parser.add_argument(
        "--allow_missing_pretrained",
        action="store_true",
        help="允许本地预训练权重不存在时继续训练；正式实验不建议开启。",
    )
    parser.add_argument(
        "--use_timm_pretrained",
        action="store_true",
        help="忽略本地权重路径，直接使用 timm pretrained=True 在线下载权重。",
    )
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(PARAMS)
    exp_name = args.experiment or cfg["active_experiment"]
    cfg.update(EXPERIMENTS[exp_name])
    cfg["active_experiment"] = exp_name

    for key in [
        "train_dir", "val_dir", "root_save_path", "epochs", "batch_size",
        "lr", "img_size", "local_weight_path",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if getattr(args, "allow_missing_pretrained", False):
        cfg["require_pretrained"] = False

    if getattr(args, "use_timm_pretrained", False):
        cfg["pretrained"] = True
        cfg["require_pretrained"] = False
        cfg["local_weight_path"] = ""

    return cfg


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def broadcast_tensor_from_rank0(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.broadcast(tensor, src=0)
    return tensor


def safe_json_dump(obj: Any, path: str) -> None:
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=4, ensure_ascii=False)


# =========================================================
# 4. 数据集与增强：四个模型共用同一套增强逻辑
# =========================================================
class AlbumentationsImageFolder(datasets.ImageFolder):
    """让 torchvision ImageFolder 使用 Albumentations，并额外返回图片路径。"""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"读取图片失败: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target, path


class SandDust(A.ImageOnlyTransform):
    """轻量沙尘模拟：暖色雾化 + 轻微噪声 + 轻微模糊。

    这里不做几何变换，也不会改变箭头方向。强度由 PARAMS 控制。
    """

    def __init__(
        self,
        alpha_limit=(0.08, 0.22),
        blur_limit=(3, 7),
        noise_std=(2.0, 10.0),
        p=0.5,
    ):
        super().__init__(p=p)
        self.alpha_limit = tuple(alpha_limit)
        self.blur_limit = tuple(blur_limit)
        self.noise_std = tuple(noise_std)

    def apply(self, img, alpha=0.15, blur_ksize=3, noise_sigma=5.0, **params):
        work = img.astype(np.float32)

        # RGB 暖沙色覆盖层。只做轻量混合，避免完全抹掉红/蓝等类别颜色。
        sand_color = np.array([205.0, 175.0, 125.0], dtype=np.float32).reshape(1, 1, 3)
        work = work * (1.0 - alpha) + sand_color * alpha

        if noise_sigma > 0:
            noise = np.random.normal(0.0, noise_sigma, size=work.shape).astype(np.float32)
            work = work + noise

        work = np.clip(work, 0, 255).astype(np.uint8)
        if blur_ksize >= 3:
            work = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)
        return work

    def get_params(self):
        low, high = self.alpha_limit
        alpha = random.uniform(float(low), float(high))

        b0, b1 = int(self.blur_limit[0]), int(self.blur_limit[1])
        candidates = [k for k in range(max(1, b0), b1 + 1) if k % 2 == 1]
        blur_ksize = random.choice(candidates) if candidates else 3

        n0, n1 = self.noise_std
        noise_sigma = random.uniform(float(n0), float(n1))
        return {"alpha": alpha, "blur_ksize": blur_ksize, "noise_sigma": noise_sigma}

    def get_transform_init_args_names(self):
        return "alpha_limit", "blur_limit", "noise_std"


def discover_dataset_classes(cfg: Dict[str, Any]) -> None:
    """先读取类别目录，再创建带 25 类分类头的 timm 模型。"""
    train_probe = datasets.ImageFolder(cfg["train_dir"])
    val_probe = datasets.ImageFolder(cfg["val_dir"])

    train_classes = list(train_probe.classes)
    val_classes = list(val_probe.classes)
    if train_classes != val_classes:
        raise ValueError(f"训练集和验证集类别不一致：train={train_classes}, val={val_classes}")

    cfg["dataset_classes"] = train_classes
    cfg["num_classes"] = len(train_classes)


def make_random_rain(cfg: Dict[str, Any]):
    """兼容 Albumentations 1.x / 2.x 的 RandomRain 参数名。"""
    slant = cfg["aug_rain_slant"]
    common = dict(
        drop_length=int(cfg["aug_rain_drop_length"]),
        drop_width=int(cfg["aug_rain_drop_width"]),
        blur_value=int(cfg["aug_rain_blur_value"]),
        brightness_coefficient=float(cfg["aug_rain_brightness"]),
        p=1.0,
    )
    try:
        return A.RandomRain(slant_range=tuple(slant), **common)
    except TypeError:
        return A.RandomRain(slant_lower=int(slant[0]), slant_upper=int(slant[1]), **common)


def make_random_snow(cfg: Dict[str, Any]):
    snow_range = tuple(float(x) for x in cfg["aug_snow_point"])
    brightness = float(cfg["aug_snow_brightness"])
    try:
        return A.RandomSnow(snow_point_range=snow_range, brightness_coeff=brightness, p=1.0)
    except TypeError:
        return A.RandomSnow(
            snow_point_lower=snow_range[0],
            snow_point_upper=snow_range[1],
            brightness_coeff=brightness,
            p=1.0,
        )


def make_random_fog(cfg: Dict[str, Any]):
    fog_range = tuple(float(x) for x in cfg["aug_fog_coef"])
    alpha = float(cfg["aug_fog_alpha"])
    try:
        return A.RandomFog(fog_coef_range=fog_range, alpha_coef=alpha, p=1.0)
    except TypeError:
        return A.RandomFog(
            fog_coef_lower=fog_range[0],
            fog_coef_upper=fog_range[1],
            alpha_coef=alpha,
            p=1.0,
        )


def make_random_sunflare(cfg: Dict[str, Any]):
    img_size = int(cfg["img_size"])
    radius = max(8, int(img_size * float(cfg["aug_sunflare_radius_ratio"])))
    roi = tuple(float(x) for x in cfg["aug_sunflare_roi"])
    try:
        return A.RandomSunFlare(
            flare_roi=roi,
            num_flare_circles_range=(1, 4),
            src_radius=radius,
            p=1.0,
        )
    except TypeError:
        return A.RandomSunFlare(
            flare_roi=roi,
            angle_lower=0.0,
            angle_upper=1.0,
            num_flare_circles_lower=1,
            num_flare_circles_upper=4,
            src_radius=radius,
            p=1.0,
        )


def make_image_compression(cfg: Dict[str, Any]):
    quality = tuple(int(x) for x in cfg["aug_jpeg_quality"])
    try:
        return A.ImageCompression(quality_range=quality, compression_type="jpeg", p=1.0)
    except (TypeError, ValueError):
        # Albumentations 1.x 默认就是 JPEG，不额外传 image_type/compression_type，兼容性更好。
        return A.ImageCompression(
            quality_lower=quality[0],
            quality_upper=quality[1],
            p=1.0,
        )


def make_channel_dropout(cfg: Dict[str, Any]):
    fill = int(cfg["aug_channel_dropout_fill"])
    try:
        return A.ChannelDropout(channel_drop_range=(1, 1), fill=fill, p=1.0)
    except TypeError:
        return A.ChannelDropout(channel_drop_range=(1, 1), fill_value=fill, p=1.0)


def make_coarse_dropout(cfg: Dict[str, Any]):
    img_size = int(cfg["img_size"])
    holes = tuple(int(x) for x in cfg["aug_coarse_holes"])
    ratio = tuple(float(x) for x in cfg["aug_coarse_hole_ratio"])
    min_size = max(2, int(img_size * ratio[0]))
    max_size = max(min_size, int(img_size * ratio[1]))
    fill = int(cfg["aug_dropout_fill"])

    try:
        return A.CoarseDropout(
            num_holes_range=holes,
            hole_height_range=(min_size, max_size),
            hole_width_range=(min_size, max_size),
            fill=fill,
            p=1.0,
        )
    except TypeError:
        return A.CoarseDropout(
            min_holes=holes[0],
            max_holes=holes[1],
            min_height=min_size,
            max_height=max_size,
            min_width=min_size,
            max_width=max_size,
            fill_value=fill,
            p=1.0,
        )


def make_grid_dropout(cfg: Dict[str, Any]):
    img_size = int(cfg["img_size"])
    unit_ratio = tuple(float(x) for x in cfg["aug_grid_unit_ratio"])
    unit_min = max(4, int(img_size * unit_ratio[0]))
    unit_max = max(unit_min + 1, int(img_size * unit_ratio[1]))
    ratio = float(cfg["aug_grid_ratio"])
    fill = int(cfg["aug_dropout_fill"])

    try:
        return A.GridDropout(
            ratio=ratio,
            unit_size_range=(unit_min, unit_max),
            random_offset=True,
            fill=fill,
            p=1.0,
        )
    except TypeError:
        return A.GridDropout(
            ratio=ratio,
            unit_size_min=unit_min,
            unit_size_max=unit_max,
            random_offset=True,
            fill_value=fill,
            p=1.0,
        )


def build_physical_degradation_group(cfg: Dict[str, Any]):
    """雨雪雾沙尘、光学退化、压缩失真三类中随机选择一类。"""
    weather = A.OneOf([
        make_random_rain(cfg),
        make_random_snow(cfg),
        make_random_fog(cfg),
        SandDust(
            alpha_limit=tuple(cfg["aug_sand_alpha"]),
            blur_limit=tuple(cfg["aug_sand_blur_limit"]),
            noise_std=tuple(cfg["aug_sand_noise_std"]),
            p=1.0,
        ),
    ], p=1.0)

    optical = A.OneOf([
        A.MotionBlur(blur_limit=tuple(cfg["aug_motion_blur_limit"]), p=1.0),
        A.GaussianBlur(blur_limit=tuple(cfg["aug_gaussian_blur_limit"]), p=1.0),
        make_random_sunflare(cfg),
        A.ISONoise(
            color_shift=tuple(float(x) for x in cfg["aug_iso_color_shift"]),
            intensity=tuple(float(x) for x in cfg["aug_iso_intensity"]),
            p=1.0,
        ),
    ], p=1.0)

    compression = make_image_compression(cfg)
    return A.OneOf([weather, optical, compression], p=float(cfg["aug_physical_group_p"]))


def build_photometric_group(cfg: Dict[str, Any]):
    """颜色、低照度与局部对比度增强；每次最多选一种。"""
    return A.OneOf([
        A.HueSaturationValue(
            hue_shift_limit=int(cfg["aug_hue_shift_limit"]),
            sat_shift_limit=int(cfg["aug_sat_shift_limit"]),
            val_shift_limit=int(cfg["aug_val_shift_limit"]),
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=tuple(float(x) for x in cfg["aug_brightness_limit"]),
            contrast_limit=tuple(float(x) for x in cfg["aug_contrast_limit"]),
            p=1.0,
        ),
        A.CLAHE(
            clip_limit=tuple(float(x) for x in cfg["aug_clahe_clip_limit"]),
            tile_grid_size=(8, 8),
            p=1.0,
        ),
        make_channel_dropout(cfg),
        A.ToGray(p=1.0),
    ], p=float(cfg["aug_photometric_group_p"]))


def build_occlusion_group(cfg: Dict[str, Any]):
    """局部遮挡与污损模拟；避免使用过大遮挡块。"""
    return A.OneOf([
        make_coarse_dropout(cfg),
        make_grid_dropout(cfg),
    ], p=float(cfg["aug_occlusion_group_p"]))


def build_transforms(cfg: Dict[str, Any]):
    img_size = int(cfg["img_size"])
    mean = [float(x) for x in cfg["normalize_mean"]]
    std = [float(x) for x in cfg["normalize_std"]]

    # 交通标识方向具有类别语义，因此这里只做 Resize，不做 flip/rotation/random crop。
    train_ops = [A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC)]

    if bool(cfg["use_physical_degradation"]):
        train_ops.append(build_physical_degradation_group(cfg))
    if bool(cfg["use_photometric_aug"]):
        train_ops.append(build_photometric_group(cfg))
    if bool(cfg["use_occlusion_aug"]):
        train_ops.append(build_occlusion_group(cfg))

    train_ops.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_ops = [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_ops), A.Compose(val_ops)


def build_dataloaders(cfg: Dict[str, Any], rank: int, world_size: int):
    train_transform, val_transform = build_transforms(cfg)
    train_dataset = AlbumentationsImageFolder(cfg["train_dir"], transform=train_transform)
    val_dataset = AlbumentationsImageFolder(cfg["val_dir"], transform=val_transform)

    if list(train_dataset.classes) != list(cfg["dataset_classes"]):
        raise ValueError("训练集类别顺序在构建 DataLoader 时发生变化。")
    if list(val_dataset.classes) != list(cfg["dataset_classes"]):
        raise ValueError("验证集类别顺序与训练集不一致。")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )

    # 验证仅在 rank0 完整运行，不使用 DistributedSampler，保证指标与错分文件完整。
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )
    return train_dataset, val_dataset, train_loader, val_loader, train_sampler


# =========================================================
# 5. 模型构建：四个模型统一走 timm，但 model_arch 不同
# =========================================================
def load_weight_file(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """
    兼容常见本地权重格式：
    1) .safetensors: safetensors.torch.load_file
    2) .pth / .pt / .bin: torch.load
    3) torch checkpoint 外层可能是 model / state_dict / model_state_dict / module
    """
    if not path:
        return {}

    lower = path.lower()
    if lower.endswith(".safetensors"):
        return load_safetensors(path, device="cpu")

    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_known_prefixes(key: str) -> str:
    """反复去掉常见 wrapper 前缀，兼容 DDP / 自定义 backbone wrapper / timm checkpoint。"""
    prefixes = ["module.", "model.", "backbone.", "student.", "teacher."]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def resolve_pretrained_weight_path(weight_path: str) -> str:
    """兼容当前工作区与旧路径，自动找到可用的预训练权重。"""
    if not weight_path:
        return ""

    candidates = []
    for p in [weight_path, os.path.expanduser(weight_path)]:
        if p:
            candidates.append(p)

    basename = os.path.basename(weight_path)
    if basename:
        for base_dir in [
            "/home/ubuntuuser/classify/pretrain",
            "/home/ubuntuuser/pretrain",
            "/root/Image_Classify/pretrain",
        ]:
            candidates.append(os.path.join(base_dir, basename))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return weight_path


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    返回多个候选版本，尽量兼容 module./backbone./model. 前缀。
    保留 raw 版本，同时生成 strip_known_prefixes 后的版本。
    """
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [("raw", state_dict)]
    stripped = {strip_known_prefixes(k): v for k, v in state_dict.items()}
    if stripped.keys() != state_dict.keys():
        candidates.append(("strip_common_prefixes", stripped))

    # 兼容少数 checkpoint：键名中可能出现 features.module.xxx 这类局部 module 前缀。
    local_module_stripped = {k.replace(".module.", "."): v for k, v in stripped.items()}
    if local_module_stripped.keys() != stripped.keys():
        candidates.append(("strip_inner_module", local_module_stripped))

    return candidates


def load_pretrained_safely(
    model: nn.Module,
    weight_path: str,
    rank: int,
    require_pretrained: bool = False,
    min_match_ratio: float = 0.50,
) -> None:
    """
    安全加载本地预训练权重。

    关键原则：
    - ConvNeXtV2 的 .safetensors 和 Swin 的 .pth/.pt 都支持。
    - 只加载 key 存在且 shape 完全一致的参数，自动跳过分类头 shape mismatch。
    - require_pretrained=True 时，如果路径为空、不存在、完全不匹配或匹配比例过低，直接报错。
      这样可以避免 teacher 实际随机初始化但你没发现。
    """
    if not weight_path:
        msg = "local_weight_path 为空，不加载本地预训练权重。"
        if require_pretrained:
            raise FileNotFoundError("require_pretrained=True，但 " + msg)
        if rank == 0:
            print("ℹ️ " + msg)
        return

    if not os.path.exists(weight_path):
        msg = f"本地权重不存在: {weight_path}"
        if require_pretrained:
            raise FileNotFoundError(msg)
        if rank == 0:
            print(f"⚠️ {msg}，跳过加载。")
        return

    if rank == 0:
        print(f"✅ 尝试加载本地权重: {weight_path}")

    raw_sd = load_weight_file(weight_path)
    if not isinstance(raw_sd, dict):
        raise TypeError(f"权重文件解析结果不是 dict，无法加载: {weight_path}")

    model_sd = model.state_dict()
    best_matched: Dict[str, torch.Tensor] = {}
    best_skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    best_name = "raw"

    for name, cand in normalize_state_dict_keys(raw_sd):
        matched: Dict[str, torch.Tensor] = {}
        skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for k, v in cand.items():
            if k in model_sd and hasattr(v, "shape"):
                if model_sd[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        if len(matched) > len(best_matched):
            best_matched = matched
            best_skipped_shape = skipped_shape
            best_name = name

    match_ratio = len(best_matched) / max(len(model_sd), 1)

    if len(best_matched) == 0:
        msg = "没有匹配到任何预训练权重 key，请检查 model_arch 和权重文件是否对应。"
        if require_pretrained:
            raise RuntimeError(msg + f" weight_path={weight_path}")
        if rank == 0:
            print("⚠️ " + msg)
        return

    if require_pretrained and match_ratio < float(min_match_ratio):
        raise RuntimeError(
            "预训练权重匹配比例过低，可能是 model_arch 与权重文件不匹配。"
            f" matched={len(best_matched)}/{len(model_sd)} ({match_ratio:.2%}), "
            f"candidate={best_name}, weight_path={weight_path}"
        )

    msg = model.load_state_dict(best_matched, strict=False)

    if rank == 0:
        print(f"📊 权重加载结果: {best_name}")
        print(f"   成功加载参数数: {len(best_matched)} / 当前模型参数数: {len(model_sd)} ({match_ratio:.2%})")
        print(f"   shape 不匹配并跳过: {len(best_skipped_shape)}")
        if best_skipped_shape:
            print(f"   shape mismatch 示例: {best_skipped_shape[:5]}")
        print(f"   Missing keys after partial load: {len(msg.missing_keys)}")
        print(f"   Unexpected keys after partial load: {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 0:
            print(f"   Missing 示例: {msg.missing_keys[:8]}")
        if len(msg.unexpected_keys) > 0:
            print(f"   Unexpected 示例: {msg.unexpected_keys[:8]}")
        if require_pretrained:
            print("   ✅ require_pretrained=True，预训练权重检查通过。")


def build_model(cfg: Dict[str, Any], rank: int) -> nn.Module:
    """统一用 timm 创建四个模型，并安全加载本地预训练权重。"""
    arch = str(cfg["model_arch"])

    create_kwargs = dict(
        pretrained=bool(cfg["pretrained"]),
        num_classes=int(cfg["num_classes"]),
        drop_path_rate=float(cfg["drop_path_rate"]),
    )

    # 这些 Transformer/混合架构需要显式传入固定输入尺寸。
    if cfg.get("model_family") in {"eva02", "maxxvitv2", "swinv2", "dinov3"}:
        create_kwargs["img_size"] = int(cfg["img_size"])

    if rank == 0:
        print(f"🏗️ timm.create_model('{arch}', img_size={cfg['img_size']})")

    try:
        model = timm.create_model(arch, **create_kwargs)
    except RuntimeError as exc:
        version = getattr(timm, "__version__", "unknown")
        hint = ""
        if cfg.get("model_family") == "dinov3":
            hint = " DINOv3 通常需要 timm>=1.0.20。"
        raise RuntimeError(
            f"timm={version} 创建模型 {arch} 失败。{hint} "
            "请检查 model_arch、timm 版本和输入尺寸。"
        ) from exc

    # pretrained=True 时由 timm 自己完成权重下载，不再重复加载本地文件。
    if not bool(cfg["pretrained"]):
        weight_path = resolve_pretrained_weight_path(cfg.get("local_weight_path", ""))
        if rank == 0 and weight_path != cfg.get("local_weight_path", ""):
            print(f"🔎 已将预训练权重路径自动映射为: {weight_path}")

        load_pretrained_safely(
            model,
            weight_path,
            rank,
            require_pretrained=bool(cfg.get("require_pretrained", False)),
            min_match_ratio=float(cfg.get("min_pretrained_match_ratio", 0.50)),
        )

    # 使用当前 timm 模型预训练配置中的 mean/std，避免 MaxxViTV2 等模型被错误归一化。
    try:
        data_cfg = timm.data.resolve_model_data_config(model)
        cfg["normalize_mean"] = [float(x) for x in data_cfg.get("mean", cfg["normalize_mean"])]
        cfg["normalize_std"] = [float(x) for x in data_cfg.get("std", cfg["normalize_std"])]
        if rank == 0:
            print(f"🧮 Normalize mean={cfg['normalize_mean']} std={cfg['normalize_std']}")
    except Exception as exc:
        if rank == 0:
            print(f"⚠️ 无法读取 timm 模型预处理配置，使用 PARAMS 默认归一化参数: {exc}")

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# =========================================================
# 6. 评估输出：metrics、混淆矩阵、预测 CSV、错分文件夹
# =========================================================
def output_class_name(class_name: str, cfg: Dict[str, Any]) -> str:
    return cfg.get("class_alias_for_output", {}).get(class_name, class_name)


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + [int(x) for x in row])


def save_classification_report_csv(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for k, v in report.items():
            if isinstance(v, dict):
                writer.writerow([
                    k,
                    v.get("precision", ""),
                    v.get("recall", ""),
                    v.get("f1-score", ""),
                    v.get("support", ""),
                ])
            else:
                writer.writerow([k, v, "", "", ""])


def save_predictions_csv(
    rows: List[Dict[str, Any]],
    class_names: List[str],
    path: str,
    save_probs: bool,
) -> None:
    base_fields = [
        "sample_index", "rel_path", "image_name", "image_path",
        "true_label_id", "true_label", "pred_label_id", "pred_label", "correct",
    ]
    prob_fields = [f"prob_{c}" for c in class_names] if save_probs else []
    fields = base_fields + prob_fields

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def save_wrong_samples(
    rows: List[Dict[str, Any]],
    class_names: List[str],
    cfg: Dict[str, Any],
    out_dir: str,
) -> None:
    wrong_rows = [r for r in rows if not bool(r["correct"])]

    wrong_csv = os.path.join(out_dir, "wrong_samples.csv")
    save_predictions_csv(wrong_rows, class_names, wrong_csv, bool(cfg["save_probs_in_csv"]))

    wrong_root = os.path.join(out_dir, "wrong_by_true_class")
    os.makedirs(wrong_root, exist_ok=True)

    for c in class_names:
        folder = os.path.join(wrong_root, output_class_name(c, cfg))
        os.makedirs(folder, exist_ok=True)
        names_txt = os.path.join(folder, "wrong_names.txt")
        with open(names_txt, "w", encoding="utf-8") as f:
            for r in wrong_rows:
                if r["true_label"] == c:
                    f.write(
                        f"{r['image_name']}\ttrue={r['true_label']}\tpred={r['pred_label']}\trel_path={r['rel_path']}\n"
                    )
                    if bool(cfg["copy_wrong_images"]):
                        src = r["image_path"]
                        dst_name = f"pred_{r['pred_label']}__{r['image_name']}"
                        dst = os.path.join(folder, dst_name)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)


def save_eval_artifacts(eval_result: Dict[str, Any], cfg: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    class_names = cfg["dataset_classes"]

    metrics_json = os.path.join(out_dir, "metrics.json")
    safe_json_dump(eval_result["metrics"], metrics_json)

    save_confusion_matrix_csv(
        eval_result["confusion_matrix"],
        class_names,
        os.path.join(out_dir, "confusion_matrix.csv"),
    )

    save_classification_report_csv(
        eval_result["classification_report"],
        os.path.join(out_dir, "classification_report.csv"),
    )

    save_predictions_csv(
        eval_result["rows"],
        class_names,
        os.path.join(out_dir, "predictions.csv"),
        bool(cfg["save_probs_in_csv"]),
    )

    save_wrong_samples(eval_result["rows"], class_names, cfg, out_dir)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    class_names = cfg["dataset_classes"]
    num_classes = len(class_names)

    total_loss = 0.0
    total_steps = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_paths: List[str] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels, paths in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if bool(cfg["channels_last"]):
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            with amp_autocast(device, bool(cfg["use_amp"])):
                outputs = model(imgs)
                # 验证阶段只使用真实标签计算标准交叉熵。
                loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += float(loss.item())
            total_steps += 1
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_paths.extend(list(paths))
            all_probs.extend(probs.detach().cpu().float().numpy())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    rows: List[Dict[str, Any]] = []
    val_root = cfg["val_dir"]
    for i, (path, true_id, pred_id, prob) in enumerate(zip(all_paths, y_true, y_pred, all_probs)):
        true_label = class_names[int(true_id)]
        pred_label = class_names[int(pred_id)]
        try:
            rel_path = os.path.relpath(path, val_root)
        except ValueError:
            rel_path = path

        row = {
            "sample_index": i,
            "rel_path": rel_path,
            "image_name": os.path.basename(path),
            "image_path": path,
            "true_label_id": int(true_id),
            "true_label": true_label,
            "pred_label_id": int(pred_id),
            "pred_label": pred_label,
            "correct": int(true_id == pred_id),
        }
        if bool(cfg["save_probs_in_csv"]):
            for j, c in enumerate(class_names):
                row[f"prob_{c}"] = float(prob[j])
        rows.append(row)

    metrics = {
        "loss": total_loss / max(total_steps, 1),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "lowest_class_f1": float(min(report[c]["f1-score"] for c in class_names)),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in class_names
        },
    }

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "rows": rows,
    }


# =========================================================
# 7. checkpoint：只保存最佳推理模型，不保留断点续训分支
# =========================================================
def save_inference_checkpoint(
    path: str,
    epoch: int,
    model_to_save: nn.Module,
    best_f1: float,
    val_metrics: Dict[str, Any],
    cfg: Dict[str, Any],
    checkpoint_type: str = "ema_best_for_softvote",
) -> None:
    """保存可直接用于单模型推理和四模型 SoftVote 的 checkpoint。"""
    torch.save({
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "best_f1": best_f1,
        "val_acc": val_metrics.get("accuracy"),
        "val_loss": val_metrics.get("loss"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "model_arch": cfg.get("model_arch"),
        "img_size": cfg.get("img_size"),
        "num_classes": cfg.get("num_classes"),
        "dataset_classes": cfg.get("dataset_classes"),
        "normalize_mean": cfg.get("normalize_mean"),
        "normalize_std": cfg.get("normalize_std"),
        "config": cfg,
        "usage": "inference_or_softvote_only",
    }, path)


def append_epoch_log(log_path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(log_path)
    fields = [
        "epoch", "lr", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "val_macro_precision", "val_macro_recall", "val_lowest_class_f1",
        "best_f1", "early_stop_counter",
    ]
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


# =========================================================
# 8. 主训练流程
# =========================================================
def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    if bool(cfg.get("suppress_known_warnings", True)):
        apply_warning_filters()

    rank, world_size, local_rank, device = setup_distributed()
    seed_everything(int(cfg["seed"]), rank)

    # 先探测类别数量，再创建对应 25 类分类头。
    discover_dataset_classes(cfg)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(cfg["root_save_path"], cfg["model_tag"])
    if is_main_process(rank):
        os.makedirs(model_dir, exist_ok=True)

    # 先构建模型，以便读取模型专属 mean/std；随后再构建数据增强和 DataLoader。
    model = build_model(cfg, rank).to(device)
    if bool(cfg["channels_last"]):
        model = model.to(memory_format=torch.channels_last)

    train_dataset, val_dataset, train_loader, val_loader, train_sampler = build_dataloaders(
        cfg, rank, world_size
    )

    if is_main_process(rank):
        safe_json_dump(cfg, os.path.join(model_dir, f"config_{run_id}.json"))
        print("=" * 88)
        print(f"实验: {cfg['active_experiment']} | stage={cfg['stage']} | role={cfg['role']}")
        print(f"模型: {cfg['model_arch']} | 输入尺寸: {cfg['img_size']}")
        print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 类别数: {cfg['num_classes']}")
        print(f"类别顺序: {cfg['dataset_classes']}")
        print(f"保存目录: {model_dir} | GPU数: {world_size}")
        print(
            "增强: 物理退化/天气 OneOf + 颜色/光度 OneOf + "
            "CoarseDropout/GridDropout OneOf"
        )
        print("禁用: horizontal flip / vertical flip / rotation / random crop / Mixup / CutMix")
        print("checkpoint: 只保存最佳推理模型，不保存断点续训状态")
        print("=" * 88)

    # EMA 只在 rank0 创建，用于验证和最终 SoftVote 权重保存。
    model_ema = ModelEmaV2(model, decay=float(cfg["ema_decay"])) \
        if (bool(cfg["use_ema"]) and rank == 0) else None

    if world_size > 1 and bool(cfg["use_sync_bn"]) and device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["label_smoothing"])
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    warmup_epochs = int(cfg["warmup_epochs"])
    total_epochs = int(cfg["epochs"])
    if warmup_epochs > 0:
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=float(cfg["min_lr"]),
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            [scheduler_warmup, scheduler_cosine],
            [warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=float(cfg["min_lr"]),
        )

    scaler = make_amp_scaler(enabled=bool(cfg["use_amp"]) and device.type == "cuda")

    best_f1 = -1.0
    early_stop_counter = 0
    log_csv = os.path.join(model_dir, f"epoch_log_{run_id}.csv")

    for epoch in range(total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for imgs, labels, _paths in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if bool(cfg["channels_last"]):
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, bool(cfg["use_amp"])):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(cfg["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss_tensor = torch.tensor(
            train_loss_sum / max(train_steps, 1),
            device=device,
        )
        train_loss = reduce_mean_tensor(train_loss_tensor, world_size).item()

        # 验证仅由 rank0 完整执行。
        val_metrics_tensor = torch.zeros(5, device=device)
        eval_result = None
        if is_main_process(rank):
            eval_model = model_ema.module if model_ema is not None else unwrap_model(model)
            eval_result = evaluate(eval_model, val_loader, criterion, cfg, device)
            m = eval_result["metrics"]
            val_metrics_tensor = torch.tensor([
                m["loss"], m["accuracy"], m["macro_f1"],
                m["macro_precision"], m["macro_recall"],
            ], device=device, dtype=torch.float32)

        val_metrics_tensor = broadcast_tensor_from_rank0(val_metrics_tensor, world_size)
        val_loss, val_acc, val_f1, val_precision, val_recall = [
            float(x) for x in val_metrics_tensor.detach().cpu().tolist()
        ]

        if world_size > 1:
            dist.barrier()

        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process(rank):
            assert eval_result is not None
            lowest_class_f1 = eval_result["metrics"]["lowest_class_f1"]

            print(f"\nEpoch {epoch + 1}/{total_epochs} | LR: {current_lr:.2e}")
            print(f"TRAIN | Loss {train_loss:.4f}")
            print(
                f"VAL   | Loss {val_loss:.4f} | Acc {val_acc:.4f} | "
                f"Macro-F1 {val_f1:.4f} | Lowest-F1 {lowest_class_f1:.4f}"
            )
            print("PER-CLASS F1:")
            for c, d in eval_result["metrics"]["per_class"].items():
                print(
                    f"  {c:>12s}: P={d['precision']:.4f} R={d['recall']:.4f} "
                    f"F1={d['f1']:.4f} N={d['support']}"
                )

            improved = val_f1 > best_f1
            if improved:
                best_f1 = val_f1
                early_stop_counter = 0

                eval_dir = os.path.join(model_dir, f"best_eval_{run_id}")
                if bool(cfg["save_eval_outputs"]):
                    save_eval_artifacts(eval_result, cfg, eval_dir)

                inference_model = model_ema.module if model_ema is not None else unwrap_model(model)
                best_path = os.path.join(
                    model_dir,
                    f"best_model_{run_id}_ep{epoch + 1}_f1_{val_f1:.4f}.pth",
                )
                save_inference_checkpoint(
                    best_path,
                    epoch + 1,
                    inference_model,
                    best_f1,
                    eval_result["metrics"],
                    cfg,
                    checkpoint_type=(
                        "ema_best_for_softvote" if model_ema is not None
                        else "raw_best_for_softvote"
                    ),
                )
                print(f"✅ 最佳 SoftVote 推理模型已保存: {best_path}")
                print(f"📁 最佳验证输出已保存: {eval_dir}")
            else:
                early_stop_counter += 1
                print(f"⚠️ 早停计数 {early_stop_counter}/{cfg['patience']}")

            append_epoch_log(log_csv, {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "val_macro_precision": val_precision,
                "val_macro_recall": val_recall,
                "val_lowest_class_f1": lowest_class_f1,
                "best_f1": best_f1,
                "early_stop_counter": early_stop_counter,
            })

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if is_main_process(rank) and early_stop_counter >= int(cfg["patience"]):
            stop_flag[0] = 1
        if world_size > 1:
            dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if is_main_process(rank):
                print("🛑 早停触发")
            break

        scheduler.step()

    if is_main_process(rank):
        print(f"\n🏁 训练完成！Best Macro-F1: {best_f1:.4f}")
        print(f"📂 模型目录: {model_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
