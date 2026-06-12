# train_cvgtext_tcvd.py

import os
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


# ============================================================
# Project root
# 这个文件放在项目根目录：
# /home/ly/myproject/lw/Sample4Geo-main/train_cvgtext_tcvd.py
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Existing modules
# ============================================================
from sample4geo.dataset.cvgtext_photos import CVGTextPhotosTrain, CVGTextPhotosEval
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger

# ============================================================
# New T-CVD modules
# ============================================================
from sample4geo.model_tcvd import TextGuidedCVDModel
from sample4geo.loss_tcvd import TCVDLoss
from sample4geo.trainer_tcvd import train_tcvd, predict_tcvd

# Reuse existing evaluation function
from sample4geo.evaluate.cvusa_and_cvact import calculate_scores


@dataclass
class Configuration:
    # ============================================================
    # Dataset
    # ============================================================
    data_root: str = "/home/ly/DATA/CVG-Text_full"

    city_train: tuple = ("NewYork",)
    city_eval: tuple = ("Tokyo", "NewYork", "Brisbane")

    # ============================================================
    # Image backbone
    # ============================================================
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384

    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"

    # ============================================================
    # Text encoder
    # ============================================================
    text_model_name: str = "openai/clip-vit-large-patch14-336"
    text_checkpoint_path: str = "/home/ly/myproject/lw/text_pretrained/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth"

    freeze_text_encoder: bool = True
    strict_text_ckpt: bool = False
    text_max_len: int = 300

    # ============================================================
    # T-CVD alpha split
    #
    # content_dim = alpha * backbone_feature_dim
    # viewpoint_dim = (1 - alpha) * backbone_feature_dim
    #
    # 如果 ConvNeXt-Base 输出 feat_dim=1024:
    # alpha=0.5  -> content_dim=512, viewpoint_dim=512
    # alpha=0.75 -> content_dim=768, viewpoint_dim=256
    # ============================================================
    split_alpha: float = 0.5

    tcvd_hidden_dim: int = 2048
    tcvd_dropout: float = 0.1

    # ============================================================
    # Loss weights
    #
    # 注意：
    # 你现在 T-CVD 只有 94，基线 96，所以这里先用稳一点的配置：
    # 先开 text + iic，先不开 rec。
    # ============================================================
    lambda_txt: float = 0.01
    lambda_iic: float = 0.01
    lambda_rec: float = 0.05

    use_text_loss: bool = True
    use_iic_loss: bool = True
    use_rec_loss: bool = False

    # ============================================================
    # Optimization
    # ============================================================
    epochs: int = 80
    batch_size: int = 32
    batch_size_eval: int = 64

    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1

    mixed_precision: bool = True
    label_smoothing: float = 0.1
    clip_grad: float = 100.0

    # ============================================================
    # Runtime
    # ============================================================
    seed: int = 42
    verbose: bool = True

    gpu_ids: tuple = (0,1)

    normalize_features: bool = True
    eval_every_n_epoch: int = 1

    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # ============================================================
    # Output
    # ============================================================
    model_path: str = "./cvgtext_tcvd_runs"


def save_model(model, path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def save_curves(history, save_dir):
    if len(history) == 0:
        return

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    # ============================================================
    # Loss curve
    # ============================================================
    plt.figure(figsize=(9, 5))

    for col in ["loss_total", "loss_loc", "loss_txt", "loss_iic", "loss_rec"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], linewidth=1.8, label=col)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("T-CVD Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=180)
    plt.close()

    # ============================================================
    # Recall curve
    # ============================================================
    if "r1_mean" not in df.columns:
        return

    recall_df = df[df["r1_mean"].notna()]

    if len(recall_df) == 0:
        return

    plt.figure(figsize=(9, 5))

    city_cols = [
        col for col in recall_df.columns
        if col.startswith("r1_") and col != "r1_mean"
    ]

    for col in city_cols:
        plt.plot(
            recall_df["epoch"],
            recall_df[col],
            linewidth=1.4,
            label=col,
        )

    plt.plot(
        recall_df["epoch"],
        recall_df["r1_mean"],
        linewidth=2.2,
        label="r1_mean",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Recall@1")
    plt.title("T-CVD Recall@1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "recall_curve.png"), dpi=180)
    plt.close()


def evaluate_tcvd(
    config,
    model,
    reference_loader,
    query_loader,
    ranks=[1, 5, 10],
    step_size=1000,
    cleanup=True,
):
    """
    T-CVD evaluation.

    推理阶段不使用文本，只使用 content feature：

        query_content @ reference_content.T
    """

    print("\nExtract reference content features...")
    reference_features, reference_labels = predict_tcvd(
        config=config,
        model=model,
        dataloader=reference_loader,
    )

    print("Extract query content features...")
    query_features, query_labels = predict_tcvd(
        config=config,
        model=model,
        dataloader=query_loader,
    )

    print("Compute retrieval scores...")

    r1 = calculate_scores(
        query_features,
        reference_features,
        query_labels,
        reference_labels,
        step_size=step_size,
        ranks=ranks,
    )

    if cleanup:
        del reference_features
        del reference_labels
        del query_features
        del query_labels

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return r1


def build_model(config):
    model = TextGuidedCVDModel(
        model_name=config.model,
        pretrained=True,
        img_size=config.img_size,
        pretrained_path=config.pretrained_weight_path,

        split_alpha=config.split_alpha,

        hidden_dim=config.tcvd_hidden_dim,
        dropout=config.tcvd_dropout,

        text_model_name=config.text_model_name,
        text_checkpoint_path=config.text_checkpoint_path,
        text_max_len=config.text_max_len,
        freeze_text_encoder=config.freeze_text_encoder,
        strict_text_ckpt=config.strict_text_ckpt,
    )

    return model


def main():
    config = Configuration()

    train_cities = (
        list(config.city_train)
        if isinstance(config.city_train, (tuple, list))
        else [str(config.city_train)]
    )

    eval_cities = (
        list(config.city_eval)
        if isinstance(config.city_eval, (tuple, list))
        else [str(config.city_eval)]
    )

    city_tag = "-".join(train_cities)

    loss_tag = (
        f"alpha{config.split_alpha}_"
        f"txt{config.lambda_txt}_"
        f"iic{config.lambda_iic}_"
        f"rec{config.lambda_rec}_"
        f"useRec{int(config.use_rec_loss)}"
    )

    run_dir = os.path.join(
        config.model_path,
        city_tag,
        config.model,
        loss_tag,
        time.strftime("%Y%m%d_%H%M%S"),
    )

    os.makedirs(run_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(run_dir, "log.txt"))

    setup_system(
        seed=config.seed,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic,
    )

    print("============================================================")
    print("Text-Guided Content-Viewpoint Disentanglement")
    print("============================================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {run_dir}")
    print(f"Train cities: {train_cities}")
    print(f"Eval cities:  {eval_cities}")
    print(f"Data root:    {config.data_root}")

    print("\nCVG-Text paths:")
    print(f"- text base: {os.path.join(config.data_root, 'annotation', 'texts')}")
    print(f"- query base: {os.path.join(config.data_root, 'data', 'query')}")
    print(
        "- reference base:",
        os.path.join(
            config.data_root,
            "mnt",
            "hwfile",
            "opendatalab",
            "air",
            "linhonglin",
            "CVG-text",
            "reference",
        ),
    )

    print("\nBackbone:")
    print(f"- model: {config.model}")
    print(f"- img_size: {config.img_size}")
    print(f"- pretrained image weight: {config.pretrained_weight_path}")

    print("\nText encoder:")
    print(f"- text model: {config.text_model_name}")
    print(f"- text checkpoint: {config.text_checkpoint_path}")
    print(f"- text max len: {config.text_max_len}")
    print(f"- freeze text encoder: {config.freeze_text_encoder}")

    print("\nT-CVD alpha split:")
    print(f"- split_alpha: {config.split_alpha}")
    print("- content_dim/viewpoint_dim will be computed inside model_tcvd.py")

    print("\nT-CVD loss:")
    print(f"- lambda_txt: {config.lambda_txt}")
    print(f"- lambda_iic: {config.lambda_iic}")
    print(f"- lambda_rec: {config.lambda_rec}")
    print(f"- use_text_loss: {config.use_text_loss}")
    print(f"- use_iic_loss: {config.use_iic_loss}")
    print(f"- use_rec_loss: {config.use_rec_loss}")
    print("============================================================")

    # ============================================================
    # Build model
    # ============================================================
    model = build_model(config)

    data_cfg = model.get_config()
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    print("\nBackbone actual dims:")
    print(f"- feat_dim: {model.feat_dim}")
    print(f"- content_dim: {model.content_dim}")
    print(f"- viewpoint_dim: {model.viewpoint_dim}")

    # Satellite image size
    image_size_sat = (config.img_size, config.img_size)

    # Ground image size, following original Sample4Geo / CVG-Text setting
    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    print("\nImage size:")
    print("- satellite:", image_size_sat)
    print("- ground:", img_size_ground)
    print("- mean:", mean)
    print("- std:", std)

    if config.device == "cuda":
        print("\nCUDA is available.")
        print("GPU count:", torch.cuda.device_count())
    else:
        print("\nUsing CPU.")

    model = model.to(config.device)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        print("Using DataParallel with gpu_ids:", config.gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # ============================================================
    # Transforms
    # ============================================================
    sat_tf_train, grd_tf_train = get_transforms_train(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
    )

    sat_tf_val, grd_tf_val = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
    )

    # ============================================================
    # Train dataset
    # ============================================================
    train_dataset = CVGTextPhotosTrain(
        data_root=config.data_root,
        city=train_cities,
        transforms_query=grd_tf_train,
        transforms_reference=sat_tf_train,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("\nTrain pairs:", len(train_dataset))

    if hasattr(train_dataset, "city_stats"):
        print("Train city stats:")
        for c, st in train_dataset.city_stats.items():
            print(
                f"  {c}: total={st['total']} "
                f"matched={st['matched']} "
                f"dropped={st['dropped']}"
            )

    # ============================================================
    # Eval datasets
    # ============================================================
    eval_sets = {}

    for city in eval_cities:
        ref_eval = CVGTextPhotosEval(
            data_root=config.data_root,
            city=city,
            split="test",
            img_type="reference",
            transforms=sat_tf_val,
        )

        qry_eval = CVGTextPhotosEval(
            data_root=config.data_root,
            city=city,
            split="test",
            img_type="query",
            transforms=grd_tf_val,
        )

        ref_loader = DataLoader(
            ref_eval,
            batch_size=config.batch_size_eval,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        qry_loader = DataLoader(
            qry_eval,
            batch_size=config.batch_size_eval,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        eval_sets[city] = {
            "ref_eval": ref_eval,
            "qry_eval": qry_eval,
            "ref_loader": ref_loader,
            "qry_loader": qry_loader,
        }

        print(f"\nEval [{city}]")
        print(f"- reference: {len(ref_eval)}")
        print(f"- query:     {len(qry_eval)}")

        if hasattr(ref_eval, "stats"):
            print(
                f"- stats: total={ref_eval.stats['total']} "
                f"matched={ref_eval.stats['matched']} "
                f"dropped={ref_eval.stats['dropped']}"
            )

    # ============================================================
    # Loss / optimizer / scheduler
    # ============================================================
    loss_function = TCVDLoss(
        lambda_txt=config.lambda_txt,
        lambda_iic=config.lambda_iic,
        lambda_rec=config.lambda_rec,
        label_smoothing=config.label_smoothing,
        use_text_loss=config.use_text_loss,
        use_iic_loss=config.use_iic_loss,
        use_rec_loss=config.use_rec_loss,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
    )

    train_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=train_steps,
        num_warmup_steps=warmup_steps,
    )

    scaler = GradScaler(init_scale=2.0 ** 10) if config.mixed_precision else None

    print("\nTraining config:")
    print(f"- optimizer: AdamW")
    print(f"- lr: {config.lr}")
    print(f"- epochs: {config.epochs}")
    print(f"- batch_size: {config.batch_size}")
    print(f"- batch_size_eval: {config.batch_size_eval}")
    print(f"- mixed_precision: {config.mixed_precision}")
    print(f"- train_steps: {train_steps}")
    print(f"- warmup_steps: {warmup_steps}")
    print(f"- output dir: {run_dir}")

    # ============================================================
    # Train loop
    # ============================================================
    best_r1 = -1.0
    history = []
    interrupted = False

    try:
        for epoch in range(1, config.epochs + 1):
            print(f"\n{'-' * 30}[Epoch {epoch}/{config.epochs}]{'-' * 30}")

            losses = train_tcvd(
                config=config,
                model=model,
                dataloader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

            print(
                "Epoch {} | total={:.4f} loc={:.4f} txt={:.4f} "
                "iic={:.4f} rec={:.4f} lr={:.8f}".format(
                    epoch,
                    losses["loss_total"],
                    losses["loss_loc"],
                    losses["loss_txt"],
                    losses["loss_iic"],
                    losses["loss_rec"],
                    optimizer.param_groups[0]["lr"],
                )
            )

            r1_mean = None
            r1_by_city = {city: None for city in eval_cities}

            if (epoch % config.eval_every_n_epoch == 0) or (epoch == config.epochs):
                print("\n------------------------------[Evaluate T-CVD]------------------------------")

                r1_vals = []

                for city in eval_cities:
                    print(f"\n[Eval City] {city}")

                    r1_city = evaluate_tcvd(
                        config=config,
                        model=model,
                        reference_loader=eval_sets[city]["ref_loader"],
                        query_loader=eval_sets[city]["qry_loader"],
                        ranks=[1, 5, 10],
                        step_size=1000,
                        cleanup=True,
                    )

                    r1_by_city[city] = r1_city
                    r1_vals.append(r1_city)

                if len(r1_vals) > 0:
                    r1_mean = float(sum(r1_vals) / len(r1_vals))
                else:
                    r1_mean = None

                print(f"\nR@1 mean across eval cities: {r1_mean:.4f}")

                if r1_mean is not None and r1_mean > best_r1:
                    best_r1 = r1_mean

                    save_path = os.path.join(
                        run_dir,
                        f"weights_best_e{epoch}_{r1_mean:.4f}.pth",
                    )

                    save_model(model, save_path)
                    print("Saved best model:", save_path)

            row = {
                "epoch": epoch,

                "loss_total": losses["loss_total"],
                "loss_loc": losses["loss_loc"],
                "loss_txt": losses["loss_txt"],
                "loss_iic": losses["loss_iic"],
                "loss_rec": losses["loss_rec"],

                "r1_mean": r1_mean,
                "lr": optimizer.param_groups[0]["lr"],
            }

            for city in eval_cities:
                row[f"r1_{city}"] = r1_by_city[city]

            history.append(row)
            save_curves(history, run_dir)

    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Saving current model...")

    finally:
        last_path = os.path.join(run_dir, "weights_last_or_interrupt.pth")
        save_model(model, last_path)
        print("Saved last/interrupted model:", last_path)

        save_curves(history, run_dir)

        if not interrupted:
            final_path = os.path.join(run_dir, "weights_end.pth")
            save_model(model, final_path)
            print("Saved final model:", final_path)

        print("\nTraining finished.")
        print("Saved outputs:")
        print(f"- log:          {os.path.join(run_dir, 'log.txt')}")
        print(f"- history:      {os.path.join(run_dir, 'history.csv')}")
        print(f"- loss curve:   {os.path.join(run_dir, 'loss_curve.png')}")
        print(f"- recall curve: {os.path.join(run_dir, 'recall_curve.png')}")
        print(f"- model dir:    {run_dir}")


if __name__ == "__main__":
    main()