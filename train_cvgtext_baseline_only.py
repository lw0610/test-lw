import os
import time
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from sample4geo.dataset.cvgtext_photos import CVGTextPhotosTrain, CVGTextPhotosEval
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel
from sample4geo.trainer import train, predict
from sample4geo.evaluate.cvusa_and_cvact import evaluate


class CVGTextPhotosTrainBaseline(CVGTextPhotosTrain):
    """Strict baseline dataset: only image pair + label, no text branch."""

    def __getitem__(self, index):
        q_img, r_img, _text, label = super().__getitem__(index)
        return q_img, r_img, label


@dataclass
class Configuration:
    # Data
    data_root: str = "/home/ly/DATA/CVG-Text_full"
    # city: tuple = ("Tokyo", "NewYork", "Brisbane")
    city_train: tuple = ("NewYork",)
    city_eval: tuple = ("Tokyo", "NewYork", "Brisbane")

    # Baseline model (original Sample4Geo image-only)
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"

    # Train
    epochs: int = 80
    batch_size: int = 16
    batch_size_eval: int = 64
    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    mixed_precision: bool = True
    label_smoothing: float = 0.1
    clip_grad: float = 100.0

    # Runtime
    seed: int = 42
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True
    eval_every_n_epoch: int = 1

    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Visualization
    retrieval_k: int = 5
    vis_samples: int = 200
    save_wrong_max: int = 100

    # Output
    model_path: str = "./cvgtext_runs_baseline_only"


def save_curves(history, save_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["loss_total"], marker="o", label="loss_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Baseline Only)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=180)
    plt.close()

    recall_df = df[df["r1"].notna()]
    if len(recall_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(recall_df["epoch"], recall_df["r1"], marker="o", label="Recall@1")
        plt.xlabel("Epoch")
        plt.ylabel("Recall@1 (%)")
        plt.title("Validation Recall Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "recall_curve.png"), dpi=180)
        plt.close()


def save_topk_and_wrong_cases(config, model, reference_loader, query_loader, reference_dataset, query_dataset, save_dir):
    print("\nGenerating retrieval visualizations...")

    ref_feat, ref_labels = predict(config, model, reference_loader)
    qry_feat, qry_labels = predict(config, model, query_loader)

    sim = qry_feat @ ref_feat.T
    _, topk_ids = torch.topk(sim, k=config.retrieval_k, dim=1)

    idx2ref = {int(i): p for p, i in zip(reference_dataset.items, reference_dataset.labels)}
    idx2qry = {int(i): p for p, i in zip(query_dataset.items, query_dataset.labels)}

    topk_dir = os.path.join(save_dir, "retrieval_topk")
    wrong_dir = os.path.join(save_dir, "wrong_cases_top1")
    os.makedirs(topk_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    n_vis = min(config.vis_samples, len(qry_labels))
    wrong_saved = 0

    for i in range(len(qry_labels)):
        qid = int(qry_labels[i].item())
        pred_ids = [int(ref_labels[pos].item()) for pos in topk_ids[i].cpu().tolist()]

        is_wrong_top1 = len(pred_ids) > 0 and pred_ids[0] != qid
        need_topk_vis = i < n_vis
        need_wrong_vis = is_wrong_top1 and wrong_saved < config.save_wrong_max

        if not (need_topk_vis or need_wrong_vis):
            continue

        q_path = idx2qry.get(qid)
        gt_path = idx2ref.get(qid)
        if q_path is None or gt_path is None:
            continue

        cols = 2 + config.retrieval_k
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))

        q_img = plt.imread(q_path)
        gt_img = plt.imread(gt_path)

        axes[0].imshow(q_img)
        axes[0].set_title(f"Query\nID={qid}")
        axes[0].axis('off')

        axes[1].imshow(gt_img)
        axes[1].set_title(f"GT Ref\nID={qid}")
        axes[1].axis('off')

        for j, rid in enumerate(pred_ids, start=2):
            rank = j - 1
            r_img = plt.imread(idx2ref[rid])
            axes[j].imshow(r_img)
            axes[j].set_title(f"Top-{rank}\nID={rid}\n{'HIT' if rid == qid else 'MISS'}")
            axes[j].axis('off')

            if rid == qid:
                for spine in axes[j].spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(3)

        plt.tight_layout()
        out_name = f"query_{i:04d}_id_{qid}.png"

        if need_topk_vis:
            plt.savefig(os.path.join(topk_dir, out_name), dpi=170)

        if need_wrong_vis:
            plt.savefig(os.path.join(wrong_dir, out_name), dpi=170)
            wrong_saved += 1

        plt.close(fig)

    print(f"Saved top-k visualizations: {topk_dir} (first {n_vis} queries)")
    print(f"Saved wrong top-1 cases ({wrong_saved}): {wrong_dir}")


def main():
    config = Configuration()

    train_cities = list(config.city_train)
    eval_cities = list(config.city_eval)

    city_tag = "-".join(train_cities)
    run_dir = os.path.join(config.model_path, city_tag, config.model, time.strftime("%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(run_dir, "log.txt"))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    print("Mode: strict_baseline_only")
    print(f"Train cities: {train_cities}")
    print(f"Eval cities: {eval_cities}")
    print("Using CVG-Text_full paths:")
    print(f"- text base: {os.path.join(config.data_root, 'annotation', 'texts')}")
    print(f"- query base: {os.path.join(config.data_root, 'data', 'query')}")
    print(f"- reference base: {os.path.join(config.data_root, 'mnt', 'hwfile', 'opendatalab', 'air', 'linhonglin', 'CVG-text', 'reference')}")
    print(f"Data root: {config.data_root}")

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      pretrained_path=config.pretrained_weight_path)

    data_cfg = model.get_config()
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    image_size_sat = (config.img_size, config.img_size)
    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    sat_tf_train, grd_tf_train = get_transforms_train(image_size_sat, img_size_ground, mean=mean, std=std)
    sat_tf_val, grd_tf_val = get_transforms_val(image_size_sat, img_size_ground, mean=mean, std=std)

    train_dataset = CVGTextPhotosTrainBaseline(
        data_root=config.data_root,
        city=train_cities,
        transforms_query=grd_tf_train,
        transforms_reference=sat_tf_train,
        split="train",
    )

    eval_sets = {}
    for ec in eval_cities:
        ref_eval = CVGTextPhotosEval(
            data_root=config.data_root,
            city=ec,
            split="test",
            img_type="reference",
            transforms=sat_tf_val,
        )
        qry_eval = CVGTextPhotosEval(
            data_root=config.data_root,
            city=ec,
            split="test",
            img_type="query",
            transforms=grd_tf_val,
        )

        ref_loader = DataLoader(ref_eval,
                                batch_size=config.batch_size_eval,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

        qry_loader = DataLoader(qry_eval,
                                batch_size=config.batch_size_eval,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

        eval_sets[ec] = {
            "ref_eval": ref_eval,
            "qry_eval": qry_eval,
            "ref_loader": ref_loader,
            "qry_loader": qry_loader,
        }

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True)

    print("Train pairs:", len(train_dataset))
    if hasattr(train_dataset, 'city_stats'):
        print("Train city stats:")
        for c, st in train_dataset.city_stats.items():
            print(f"  {c}: total={st['total']} matched={st['matched']} dropped={st['dropped']}")

    for ec in eval_cities:
        ref_eval = eval_sets[ec]["ref_eval"]
        qry_eval = eval_sets[ec]["qry_eval"]
        print(f"Eval [{ec}] ref: {len(ref_eval)} qry: {len(qry_eval)}")
        if hasattr(ref_eval, 'stats'):
            print(f"Eval [{ec}] stats: total={ref_eval.stats['total']} matched={ref_eval.stats['matched']} dropped={ref_eval.stats['dropped']}")

    ce = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=ce, device=config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                 num_training_steps=train_steps,
                                                 num_warmup_steps=warmup_steps)

    scaler = GradScaler(init_scale=2.**10) if config.mixed_precision else None

    best_r1 = -1.0
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'-'*30}[Epoch: {epoch}]{'-'*30}")

        train_loss = train(config,
                           model,
                           dataloader=train_loader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print(
            "Epoch {} | total={:.4f} lr={:.6f}".format(
                epoch,
                train_loss,
                optimizer.param_groups[0]["lr"],
            )
        )

        r1_mean = None
        r1_by_city = {c: None for c in eval_cities}

        if (epoch % config.eval_every_n_epoch == 0) or (epoch == config.epochs):
            print("\n------------------------------[Evaluate]------------------------------")
            r1_vals = []
            for ec in eval_cities:
                print(f"\n[Eval City] {ec}")
                r1_city = evaluate(config=config,
                                   model=model,
                                   reference_dataloader=eval_sets[ec]["ref_loader"],
                                   query_dataloader=eval_sets[ec]["qry_loader"],
                                   ranks=[1, 5, 10],
                                   step_size=1000,
                                   cleanup=True)
                r1_by_city[ec] = r1_city
                r1_vals.append(r1_city)

            r1_mean = float(sum(r1_vals) / len(r1_vals)) if len(r1_vals) > 0 else None
            print(f"\nR@1 mean across eval cities: {r1_mean:.4f}")

            if r1_mean is not None and r1_mean > best_r1:
                best_r1 = r1_mean
                save_path = os.path.join(run_dir, f"weights_best_e{epoch}_{r1_mean:.4f}.pth")
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print("Saved best:", save_path)

        row = {
            "epoch": epoch,
            "loss_total": train_loss,
            "r1": r1_mean,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for ec in eval_cities:
            row[f"r1_{ec}"] = r1_by_city[ec]

        history.append(row)

    final_path = os.path.join(run_dir, "weights_end.pth")
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)

    save_curves(history, run_dir)

    vis_city = eval_cities[0]
    save_topk_and_wrong_cases(config,
                              model,
                              eval_sets[vis_city]["ref_loader"],
                              eval_sets[vis_city]["qry_loader"],
                              eval_sets[vis_city]["ref_eval"],
                              eval_sets[vis_city]["qry_eval"],
                              run_dir)

    print("Saved final:", final_path)
    print("Saved outputs:")
    print(f"- {run_dir}/log.txt")
    print(f"- {run_dir}/history.csv")
    print(f"- {run_dir}/loss_curve.png")
    print(f"- {run_dir}/recall_curve.png")
    print(f"- {run_dir}/retrieval_topk/*.png (visualized on eval city: {vis_city})")
    print(f"- {run_dir}/wrong_cases_top1/*.png (max {config.save_wrong_max}, eval city: {vis_city})")


if __name__ == '__main__':
    main()
