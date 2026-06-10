"""
CVG-Text 上 Sample4Geo 图像基线推理（与 eval_cvusa.py 相同流程）.

- 模型: sample4geo.model.TimmModel（单塔图像编码，无文本）
- 权重二选一:
  1) pretrained_weight_path: 仅初始化 backbone（pretrained/model.safetensors）
  2) checkpoint_start: 完整 TimmModel 权重（在 CVG-Text/CVUSA 等上训练后的 .pth，推荐用于报 Recall）
- 数据: CVGTextPhotosEval，地面 query + 卫星 reference，与 train strict_baseline_only 一致
- 指标: evaluate() -> predict() 提特征 + 余弦相似度 -> Recall@1/5/10

多模态改进请用 train_cvgtext.py；勿把 MultiModalGeoModel 的 .pth 填到本脚本的 checkpoint_start。
"""
import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.cvgtext_photos import CVGTextPhotosEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    data_root: str = "/home/ly/DATA/CVG-Text_full"
    city: str = "NewYork"

    # CVG-Text_full layout — change `city` only to run Tokyo / Brisbane / NewYork, etc.
    text_base_dir: str = "/home/ly/DATA/CVG-Text_full/annotation/texts"
    query_base_dir: str = "/home/ly/DATA/CVG-Text_full/data/query"
    reference_base_dir: str = "/home/ly/DATA/CVG-Text_full/mnt/hwfile/opendatalab/air/linhonglin/CVG-text/reference"

    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384

    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True

    # Image backbone init (same as train strict_baseline_only)
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"
    # Optional: baseline checkpoint from train (strict_baseline_only) or compatible TimmModel weights
    checkpoint_start: str = ""

    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = Configuration()


if __name__ == "__main__":
    text_path = os.path.join(config.text_base_dir, config.city, "test.json")
    query_dir = os.path.join(config.query_base_dir, f"{config.city}-ground")
    reference_dir = os.path.join(config.reference_base_dir, f"{config.city}-satellite")

    print("\n[Baseline image-only eval]")
    print("Model:", config.model)
    print("City:", config.city)
    print("Test text json:", text_path)
    print("Query ground dir:", query_dir)
    print("Reference satellite dir:", reference_dir)

    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"Missing test.json: {text_path}")
    if not os.path.isdir(query_dir):
        raise FileNotFoundError(f"Missing query dir: {query_dir}")
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"Missing reference dir: {reference_dir}")

    model = TimmModel(
        config.model,
        pretrained=True,
        img_size=config.img_size,
        pretrained_path=config.pretrained_weight_path,
    )

    data_cfg = model.get_config()
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    image_size_sat = (config.img_size, config.img_size)
    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    if config.checkpoint_start:
        if not os.path.isfile(config.checkpoint_start):
            raise FileNotFoundError(f"checkpoint_start not found: {config.checkpoint_start}")
        print("Load Sample4Geo TimmModel checkpoint:", config.checkpoint_start)
        sd = torch.load(config.checkpoint_start, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("  missing keys:", len(missing), "(first 5)", missing[:5])
        if unexpected:
            print("  unexpected keys:", len(unexpected), "(first 5)", unexpected[:5])
        if unexpected and any("heads" in k or "text" in k for k in unexpected[:20]):
            print("  [WARN] unexpected keys look like MultiModalGeoModel — use a strict_baseline_only .pth here.")
    else:
        print("No checkpoint_start: using only pretrained_weight_path (backbone init, not full retrieval weights).")

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)
    model.eval()

    sat_tf, grd_tf = get_transforms_val(image_size_sat, img_size_ground, mean=mean, std=std)

    ref_dataset = CVGTextPhotosEval(
        data_root=config.data_root,
        city=config.city,
        split="test",
        img_type="reference",
        transforms=sat_tf,
        text_base_dir=config.text_base_dir,
        query_base_dir=config.query_base_dir,
        reference_base_dir=config.reference_base_dir,
    )

    qry_dataset = CVGTextPhotosEval(
        data_root=config.data_root,
        city=config.city,
        split="test",
        img_type="query",
        transforms=grd_tf,
        text_base_dir=config.text_base_dir,
        query_base_dir=config.query_base_dir,
        reference_base_dir=config.reference_base_dir,
    )

    ref_loader = DataLoader(
        ref_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    qry_loader = DataLoader(
        qry_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Reference:", len(ref_dataset), "Query:", len(qry_dataset))
    print("\n{}[{} baseline]{}".format(30 * "-", f"CVG-Text {config.city}", 30 * "-"))

    _ = evaluate(
        config=config,
        model=model,
        reference_dataloader=ref_loader,
        query_dataloader=qry_loader,
        ranks=[1, 5, 10],
        step_size=1000,
        cleanup=True,
    )