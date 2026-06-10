import os
import time
import sys
import json
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
from sample4geo.model_multimodal import MultiModalGeoModel
from sample4geo.trainer import train, predict
from sample4geo.trainer_multimodal import train_multimodal
from sample4geo.evaluate.cvusa_and_cvact import evaluate


def evaluate_with_text_fusion(config, model, qry_loader, ref_loader, lambda_score, eval_city):
    """Evaluate with fused query feature and optional score fusion."""
    model.eval()
    core_model = model.module if hasattr(model, "module") else model

    # reference image features
    ref_feats, ref_labels = predict(config, model, ref_loader)
    ref_feats = ref_feats.to(config.device).float()
    ref_labels = ref_labels.to(config.device)

    qry_img_feats = []
    qry_fused_feats = []
    qry_labels = []
    cos_list = []
    delta_list = []

    # Build label->text map from the SAME eval city (prefer test split, fallback to train).
    test_json = os.path.join(config.data_root, "annotation", "texts", eval_city, "test.json")
    train_json = os.path.join(config.data_root, "annotation", "texts", eval_city, "train.json")

    json_path = test_json if os.path.exists(test_json) else train_json
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No text annotation found for eval city '{eval_city}': {test_json} or {train_json}")

    with open(json_path, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    print(f"[Fusion Debug][{eval_city}] test_json={test_json}")
    print(f"[Fusion Debug][{eval_city}] train_json={train_json}")
    print(f"[Fusion Debug][{eval_city}] json_path={json_path}")
    print(f"[Fusion Debug][{eval_city}] text_map_len={len(text_map)}")

    names_sorted = sorted(list(text_map.keys()))
    label2text = {i: text_map[n] for i, n in enumerate(names_sorted)}

    ds = qry_loader.dataset
    has_items = hasattr(ds, "items")
    has_labels = hasattr(ds, "labels")
    print(f"[Fusion Debug][{eval_city}] dataset_has_items={has_items}")
    print(f"[Fusion Debug][{eval_city}] dataset_has_labels={has_labels}")

    checked_count = 0
    basename_in_text_map_count = 0
    old_mapping_match_count = 0
    old_mapping_mismatch_count = 0

    if has_items and has_labels:
        n_dbg = min(10, len(ds.items), len(ds.labels))
        print(f"[Fusion Debug][{eval_city}] sample_check_count={n_dbg}")
        for i in range(n_dbg):
            q_path = ds.items[i]
            label_val = int(ds.labels[i])
            basename = os.path.basename(q_path)
            basename_in_map = basename in text_map

            old_key = names_sorted[label_val] if 0 <= label_val < len(names_sorted) else "<out_of_range>"
            old_text = label2text.get(label_val, "")
            base_text = text_map.get(basename, "")
            is_match = (basename == old_key)

            checked_count += 1
            basename_in_text_map_count += int(basename_in_map)
            old_mapping_match_count += int(is_match)
            old_mapping_mismatch_count += int(not is_match)

            print(f"[Fusion Debug][{eval_city}] idx={i} label={label_val}")
            print(f"  query_path={q_path}")
            print(f"  query_basename={basename}")
            print(f"  basename_in_text_map={basename_in_map}")
            print(f"  old_label2text_key={old_key}")
            print(f"  basename_equals_old_key={is_match}")
            print(f"  old_text_head={str(old_text)[:120]}")
            print(f"  basename_text_head={str(base_text)[:120]}")
            if not is_match:
                print("[WARNING] label2text mapping may be wrong: label does not correspond to query basename.")

        print(f"[Fusion Debug][{eval_city}] checked_count={checked_count}")
        print(f"[Fusion Debug][{eval_city}] basename_in_text_map_count={basename_in_text_map_count}")
        print(f"[Fusion Debug][{eval_city}] old_mapping_match_count={old_mapping_match_count}")
        print(f"[Fusion Debug][{eval_city}] old_mapping_mismatch_count={old_mapping_mismatch_count}")

        if old_mapping_mismatch_count > 0:
            print("[Fusion Debug][WARNING] 当前 label2text = enumerate(sorted(text_map.keys())) 的方式不可靠，应该改成根据 qry_loader.dataset.items 和 qry_loader.dataset.labels 建立 label -> text 的映射。")

    with torch.no_grad():
        for q_img, labels in qry_loader:
            q_img = q_img.to(config.device)
            sem_q = core_model.encode_semantic(q_img)
            texts = [label2text.get(int(lb.item()), "") for lb in labels]
            tokens = core_model.text_encoder.tokenize(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(config.device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(config.device)

            out_q = core_model(q_img, q_img, texts=tokens)
            fused_q = out_q["fused_q"]

            batch_cos = (fused_q * sem_q).sum(dim=-1)
            batch_delta = (fused_q - sem_q).norm(dim=-1)
            cos_list.append(batch_cos.detach().cpu())
            delta_list.append(batch_delta.detach().cpu())

            qry_img_feats.append(sem_q.detach().float())
            qry_fused_feats.append(fused_q.detach().float())
            qry_labels.append(labels.to(config.device))

    qry_img_feats = torch.cat(qry_img_feats, dim=0).to(config.device)
    qry_fused_feats = torch.cat(qry_fused_feats, dim=0).to(config.device)
    qry_labels = torch.cat(qry_labels, dim=0).to(config.device)

    cos_all = torch.cat(cos_list, dim=0)
    delta_all = torch.cat(delta_list, dim=0)
    cos_qf_g_mean = cos_all.mean().item()
    delta_mean = delta_all.mean().item()
    print(f"[Fusion Diagnose][{eval_city}] cos(fused_q, sem_q)={cos_qf_g_mean:.4f}, delta={delta_mean:.4f}")

    s_img = qry_img_feats @ ref_feats.T
    s_fused = qry_fused_feats @ ref_feats.T
    s_final = s_fused if lambda_score <= 0 else (s_fused + lambda_score * s_img)

    top1 = torch.argmax(s_final, dim=1)
    pred_labels = ref_labels[top1]
    r1 = (pred_labels == qry_labels).float().mean().item() * 100.0
    return r1, cos_qf_g_mean, delta_mean


@dataclass
class Configuration:
    # =========================
    # Dataset settings
    # =========================
    data_root: str = "/home/ly/DATA/CVG-Text_full"  # CVG-Text_full root
    city_train: tuple = ("NewYork",)  # training cities
    city_eval: tuple = ("Tokyo", "NewYork", "Brisbane")  # evaluation cities

    # =========================
    # Image backbone settings
    # =========================
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384
    sem_dim: int = 768# semantic feature dim used for retrieval and text alignment
    sty_dim: int = 512  # style feature dim used by disentangle branch
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"

    # =========================
    # Text encoder settings
    # =========================
    text_model_name: str = "openai/clip-vit-large-patch14-336"  # CLIP tokenizer/text tower pair
    text_checkpoint_path: str = "/home/ly/myproject/lw/text_pretrained/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth"  # CrossText2Loc checkpoint path
    freeze_text_encoder: bool = True  # freeze text tower in early-stage fusion training
    strict_text_ckpt: bool = False  # strict=True will fail if keys do not perfectly match
    text_max_len: int = 300  # long text length

    # =========================
    # Training mode / losses
    # =========================
    train_mode: str = "text_fusion"  # strict_baseline_only | text_fusion
    lambda_fusion: float = 1.0  # fusion query alignment loss weight
    lambda_ts: float = 0.05  # weak text-satellite alignment loss weight
    lambda_adv: float = 0.01  # modality adversarial loss weight
    lambda_rec: float = 0.1  # semantic reconstruction/consistency weight
    adv_grl_lambda: float = 1.0  # gradient reversal strength
    lambda_orth: float = 0.0  # kept for compatibility, disabled by default
    # Inference score fusion: S_final = S_img + lambda_score * S_fused
    # Set a single value here; adjust manually for ablations.
    lambda_score: float = 1

    # =========================
    # Optimization settings
    # =========================
    epochs: int = 80
    batch_size: int = 16
    batch_size_eval: int = 64
    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    mixed_precision: bool = True
    label_smoothing: float = 0.1
    clip_grad: float = 100.0

    # =========================
    # Runtime settings
    # =========================
    seed: int = 42
    # Keep progress bar visible; refresh rate is throttled in trainer.
    verbose: bool = True
    gpu_ids: tuple = (0,)  # after CUDA_VISIBLE_DEVICES remap, single-card uses (0,)
    normalize_features: bool = True
    eval_every_n_epoch: int = 1

    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # =========================
    # Visualization settings
    # =========================
    retrieval_k: int = 5
    vis_samples: int = 200
    save_wrong_max: int = 100

    # =========================
    # Output settings
    # =========================
    model_path: str = "./cvgtext_runs"


def save_curves(history, save_dir):
    if len(history) == 0:
        return

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["loss_total"], color="#1f77b4", linewidth=1.8, label="loss_total")
    if "loss_retr" in df.columns:
        plt.plot(df["epoch"], df["loss_retr"], color="#ff7f0e", linewidth=1.8, label="loss_retr")
    if "loss_txt" in df.columns:
        plt.plot(df["epoch"], df["loss_txt"], color="#2ca02c", linewidth=1.8, label="loss_txt")
    if "loss_ts" in df.columns:
        plt.plot(df["epoch"], df["loss_ts"], color="#17becf", linewidth=1.8, label="loss_ts")
    if "loss_orth" in df.columns:
        plt.plot(df["epoch"], df["loss_orth"], color="#d62728", linewidth=1.8, label="loss_orth")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=180)
    plt.close()

    recall_df = df[df["r1"].notna()] if "r1" in df.columns else pd.DataFrame()
    if len(recall_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(recall_df["epoch"], recall_df["r1"], color="#9467bd", linewidth=1.8, label="Recall@1")
        for c in [col for col in recall_df.columns if col.startswith("r1_")]:
            plt.plot(recall_df["epoch"], recall_df[c], linewidth=1.2, alpha=0.9, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Recall@1 (%)")
        plt.title("Validation Recall Curve")
        plt.grid(alpha=0.25)
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


def save_model(model, config, path):
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def build_model(config):
    if config.train_mode == "strict_baseline_only":
        model = TimmModel(
            config.model,
            pretrained=True,
            img_size=config.img_size,
            pretrained_path=config.pretrained_weight_path,
        )
    else:
        ckpt = config.text_checkpoint_path if len(config.text_checkpoint_path) > 0 else None
        model = MultiModalGeoModel(
            model_name=config.model,
            pretrained=True,
            img_size=config.img_size,
            pretrained_path=config.pretrained_weight_path,
            sem_dim=config.sem_dim,
            sty_dim=config.sty_dim,
            text_max_len=config.text_max_len,
            text_model_name=config.text_model_name,
            text_checkpoint_path=ckpt,
            freeze_text_encoder=config.freeze_text_encoder,
            strict_text_ckpt=config.strict_text_ckpt,
            fusion_hidden_dim=config.sem_dim,
        )
    return model


def main():
    # All experiment hyper-parameters are configured in `Configuration` above.
    config = Configuration()

    train_cities = list(config.city_train) if isinstance(config.city_train, (tuple, list)) else [str(config.city_train)]
    eval_cities = list(config.city_eval) if isinstance(config.city_eval, (tuple, list)) else [str(config.city_eval)]

    city_tag = "-".join(train_cities)
    run_dir = os.path.join(config.model_path, city_tag, config.model, time.strftime("%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(run_dir, "log.txt"))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    print(f"Mode: {config.train_mode}")
    print(f"Train cities: {train_cities}")
    print(f"Eval cities: {eval_cities}")
    print(f"Text model: {config.text_model_name}")
    print(f"Text checkpoint: {config.text_checkpoint_path if len(config.text_checkpoint_path) > 0 else 'None'}")
    print(f"Freeze text encoder: {config.freeze_text_encoder}")
    print(f"lambda_fusion: {config.lambda_fusion}")
    print(f"lambda_ts: {config.lambda_ts}")
    print(f"lambda_adv: {config.lambda_adv}")
    print(f"lambda_rec: {config.lambda_rec}")
    print(f"adv_grl_lambda: {config.adv_grl_lambda}")
    print("Using CVG-Text_full paths:")
    print(f"- text base: {os.path.join(config.data_root, 'annotation', 'texts')}")
    print(f"- query base: {os.path.join(config.data_root, 'data', 'query')}")
    print(f"- reference base: {os.path.join(config.data_root, 'mnt', 'hwfile', 'opendatalab', 'air', 'linhonglin', 'CVG-text', 'reference')}")
    print(f"Data root: {config.data_root}")

    model = build_model(config)

    if config.train_mode != "strict_baseline_only":
        model.text_encoder.inspect_checkpoint(config.text_checkpoint_path) if len(config.text_checkpoint_path) > 0 else None

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

    train_dataset = CVGTextPhotosTrain(
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
    retrieval_loss_fn = InfoNCE(loss_function=ce, device=config.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    train_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                 num_training_steps=train_steps,
                                                 num_warmup_steps=warmup_steps)

    scaler = GradScaler(init_scale=2.**10) if config.mixed_precision else None

    best_r1 = -1.0
    history = []

    interrupted = False

    try:
        for epoch in range(1, config.epochs + 1):
            print(f"\n{'-'*30}[Epoch: {epoch}]{'-'*30}")

            if config.train_mode == "strict_baseline_only":
                train_loss = train(config,
                                   model,
                                   dataloader=train_loader,
                                   loss_function=retrieval_loss_fn,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   scaler=scaler)

                losses = {
                    "loss_total": train_loss,
                    "loss_retr": train_loss,
                    "loss_txt": 0.0,
                    "loss_ts": 0.0,
                    "loss_adv": 0.0,
                    "loss_rec": 0.0,
                    "loss_orth": 0.0,
                }
            else:
                losses = train_multimodal(config,
                                          model,
                                          train_loader,
                                          retrieval_loss_fn,
                                          optimizer,
                                          scheduler=scheduler,
                                          scaler=scaler)

            print(
                "Epoch {} | total={:.4f} retr={:.4f} fuse={:.4f} ts={:.4f} adv={:.4f} rec={:.4f} orth={:.4f} lr={:.6f}".format(
                    epoch,
                    losses["loss_total"],
                    losses["loss_retr"],
                    losses["loss_txt"],
                    losses["loss_ts"],
                    losses["loss_adv"],
                    losses["loss_rec"],
                    losses["loss_orth"],
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

                    # Image-only baseline path (lambda_score=0)
                    r1_city_img = evaluate(config=config,
                                           model=model,
                                           reference_dataloader=eval_sets[ec]["ref_loader"],
                                           query_dataloader=eval_sets[ec]["qry_loader"],
                                           ranks=[1, 5, 10],
                                           step_size=1000,
                                           cleanup=True)

                    # Optional single score-level fusion when text branch is enabled
                    if config.train_mode == "text_fusion" and config.lambda_score > 0:
                        r1_fused, cos_qf_g_mean, delta_mean = evaluate_with_text_fusion(
                            config=config,
                            model=model,
                            qry_loader=eval_sets[ec]["qry_loader"],
                            ref_loader=eval_sets[ec]["ref_loader"],
                            lambda_score=config.lambda_score,
                            eval_city=ec,
                        )
                        print(f"[Fusion][{ec}] lambda_score={config.lambda_score}, R@1={r1_fused:.4f} (image-only={r1_city_img:.4f})")
                        print(f"[Fusion Diagnose][{ec}] cos(fused_q, sem_q)={cos_qf_g_mean:.4f}, delta={delta_mean:.4f}")

                    # Keep main tracked score as image-only for strict comparability
                    r1_city = r1_city_img

                    r1_by_city[ec] = r1_city
                    r1_vals.append(r1_city)

                r1_mean = float(sum(r1_vals) / len(r1_vals)) if len(r1_vals) > 0 else None
                print(f"\nR@1 mean across eval cities: {r1_mean:.4f}")

                if r1_mean is not None and r1_mean > best_r1:
                    best_r1 = r1_mean
                    save_path = os.path.join(run_dir, f"weights_best_e{epoch}_{r1_mean:.4f}.pth")
                    save_model(model, config, save_path)
                    print("Saved best:", save_path)

            row = {
                "epoch": epoch,
                "loss_total": losses["loss_total"],
                "loss_retr": losses["loss_retr"],
                "loss_txt": losses["loss_txt"],
                "loss_ts": losses["loss_ts"],
                "loss_adv": losses["loss_adv"],
                "loss_rec": losses["loss_rec"],
                "loss_orth": losses["loss_orth"],
                "r1": r1_mean,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for ec in eval_cities:
                row[f"r1_{ec}"] = r1_by_city[ec]
            history.append(row)

            # save curves every epoch (supports interrupted runs)
            save_curves(history, run_dir)

    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Saving intermediate outputs...")

    finally:
        if len(history) > 0:
            last_path = os.path.join(run_dir, "weights_last_or_interrupt.pth")
            save_model(model, config, last_path)
            print("Saved last/interrupted:", last_path)

        save_curves(history, run_dir)

        if not interrupted:
            final_path = os.path.join(run_dir, "weights_end.pth")
            save_model(model, config, final_path)
            print("Saved final:", final_path)

            vis_city = eval_cities[0]
            save_topk_and_wrong_cases(config,
                                      model,
                                      eval_sets[vis_city]["ref_loader"],
                                      eval_sets[vis_city]["qry_loader"],
                                      eval_sets[vis_city]["ref_eval"],
                                      eval_sets[vis_city]["qry_eval"],
                                      run_dir)

        print("Saved outputs:")
        print(f"- {run_dir}/log.txt")
        print(f"- {run_dir}/history.csv")
        print(f"- {run_dir}/loss_curve.png")
        print(f"- {run_dir}/recall_curve.png")


if __name__ == '__main__':
    main()
