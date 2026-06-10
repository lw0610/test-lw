import os
import time
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from sample4geo.dataset.cvgtext_photos import CVGTextPhotosTrain, CVGTextPhotosEval
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger, AverageMeter
from sample4geo.loss import InfoNCE
from sample4geo.trainer import predict
from sample4geo.model_rawfusion import RawFusionGeoModel


def calculate_r1_from_scores(scores, query_labels, reference_labels):
    top1 = torch.argmax(scores, dim=1)
    pred_labels = reference_labels[top1]
    return (pred_labels == query_labels).float().mean().item() * 100.0


def build_label_to_text(config, qry_dataset, eval_city):
    text_path = os.path.join(config.data_root, "annotation", "texts", eval_city, "test.json")
    with open(text_path, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    label2text = {}
    if hasattr(qry_dataset, "items") and hasattr(qry_dataset, "labels"):
        for path, label in zip(qry_dataset.items, qry_dataset.labels):
            name = os.path.basename(path)
            label2text[int(label)] = text_map.get(name, "")
    else:
        names_sorted = sorted(list(text_map.keys()))
        label2text = {i: text_map[n] for i, n in enumerate(names_sorted)}
    return label2text


def evaluate_rawfusion(config, model, qry_loader, ref_loader, eval_city):
    model.eval()
    core_model = model.module if hasattr(model, "module") else model

    ref_feats, ref_labels = predict(config, model, ref_loader)
    ref_feats = ref_feats.to(config.device).float()
    ref_labels = ref_labels.to(config.device)

    label2text = build_label_to_text(config, qry_loader.dataset, eval_city)

    qry_img_feats = []
    qry_fused_feats = []
    qry_labels = []

    with torch.no_grad():
        for q_img, labels in qry_loader:
            q_img = q_img.to(config.device)
            labels_device = labels.to(config.device)
            img_q = core_model.encode_image(q_img)

            texts = [label2text.get(int(lb.item()), "") for lb in labels]
            tokens = core_model.text_encoder.tokenize(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(config.device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(config.device)

            out = core_model(q_img, q_img, texts=tokens, adv_grl_lambda=config.adv_grl_lambda)
            fused_q = out["fused_q"]

            qry_img_feats.append(img_q.detach().float())
            qry_fused_feats.append(fused_q.detach().float())
            qry_labels.append(labels_device)

    qry_img_feats = torch.cat(qry_img_feats, dim=0).to(config.device)
    qry_fused_feats = torch.cat(qry_fused_feats, dim=0).to(config.device)
    qry_labels = torch.cat(qry_labels, dim=0).to(config.device)

    s_img = qry_img_feats @ ref_feats.T
    s_fused = qry_fused_feats @ ref_feats.T
    s_final = s_fused + config.lambda_score * s_img

    r1_img = calculate_r1_from_scores(s_img, qry_labels, ref_labels)
    r1_fused = calculate_r1_from_scores(s_fused, qry_labels, ref_labels)
    r1_final = calculate_r1_from_scores(s_final, qry_labels, ref_labels)
    img_fused_cos = F.cosine_similarity(qry_img_feats, qry_fused_feats, dim=-1).mean().item()

    print(f"[Raw Image-only][{eval_city}] R@1={r1_img:.4f}")
    print(f"[Raw Fusion-only][{eval_city}] R@1={r1_fused:.4f}")
    print(f"[Raw Score Fusion][{eval_city}] lambda_score={config.lambda_score}, R@1={r1_final:.4f}")
    print(f"[Raw Image-Fused Cosine][{eval_city}] mean={img_fused_cos:.6f}")
    return r1_img, r1_fused, r1_final, img_fused_cos


def train_rawfusion(config, model, dataloader, retrieval_loss_fn, optimizer, scheduler=None, scaler=None):
    model.train()
    meters = {k: AverageMeter() for k in ["total", "retr", "fusion", "ts", "adv", "rec"]}
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(dataloader, total=len(dataloader), mininterval=10.0) if config.verbose else dataloader

    for query, reference, texts, _ in bar:
        query = query.to(config.device)
        reference = reference.to(config.device)
        core_model = model.module if hasattr(model, "module") else model
        tokenized = core_model.text_encoder.tokenize(list(texts))
        if isinstance(tokenized, dict):
            tokenized = {k: v.to(config.device) for k, v in tokenized.items()}
        else:
            tokenized = tokenized.to(config.device)

        def forward_loss():
            out = model(query, reference, texts=tokenized, adv_grl_lambda=config.adv_grl_lambda)
            img_q = out["img_q"]
            img_g = out["img_g"]
            text_emb = out["text_emb"]
            img_q_ebr = out["img_q_ebr"]
            text_emb_ebr = out["text_emb_ebr"]
            fused_q = out["fused_q"]
            disc_logits = out["disc_logits"]
            logit_scale = model.module.logit_scale.exp() if hasattr(model, "module") else model.logit_scale.exp()

            loss_retr = retrieval_loss_fn(img_q, img_g, logit_scale)
            loss_fusion = retrieval_loss_fn(fused_q, img_g, logit_scale)
            loss_ts = retrieval_loss_fn(text_emb, img_g.detach(), logit_scale)
            loss_rec = 0.5 * (
                1.0 - (img_q_ebr * img_q).sum(dim=-1).mean()
                + 1.0 - (text_emb_ebr * text_emb).sum(dim=-1).mean()
            )

            loss_adv = torch.zeros((), device=query.device)
            if disc_logits is not None:
                bsz = img_q.shape[0]
                labels = torch.cat([
                    torch.zeros(bsz, dtype=torch.long, device=query.device),
                    torch.ones(bsz, dtype=torch.long, device=query.device),
                ], dim=0)
                loss_adv = F.cross_entropy(disc_logits, labels)

            loss = (
                loss_retr
                + config.lambda_fusion * loss_fusion
                + config.lambda_ts * loss_ts
                + config.lambda_adv * loss_adv
                + config.lambda_rec * loss_rec
            )
            return loss, loss_retr, loss_fusion, loss_ts, loss_adv, loss_rec

        if scaler:
            with autocast():
                loss, loss_retr, loss_fusion, loss_ts, loss_adv, loss_rec = forward_loss()
            scaler.scale(loss).backward()
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_retr, loss_fusion, loss_ts, loss_adv, loss_rec = forward_loss()
            loss.backward()
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and config.scheduler in ["polynomial", "cosine", "constant"]:
            scheduler.step()

        meters["total"].update(loss.item())
        meters["retr"].update(loss_retr.item())
        meters["fusion"].update(loss_fusion.item())
        meters["ts"].update(loss_ts.item())
        meters["adv"].update(loss_adv.item())
        meters["rec"].update(loss_rec.item())

        if config.verbose:
            bar.set_postfix(ordered_dict={
                "total": f"{meters['total'].avg:.4f}",
                "retr": f"{meters['retr'].avg:.4f}",
                "fuse": f"{meters['fusion'].avg:.4f}",
                "ts": f"{meters['ts'].avg:.4f}",
                "adv": f"{meters['adv'].avg:.4f}",
                "rec": f"{meters['rec'].avg:.4f}",
            })

    if config.verbose:
        bar.close()

    return {f"loss_{k}": v.avg for k, v in meters.items()}


def save_model(model, config, path):
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


@dataclass
class Configuration:
    data_root: str = "/home/ly/DATA/CVG-Text_full"
    city_train: tuple = ("NewYork",)
    city_eval: tuple = ("Tokyo", "NewYork", "Brisbane")

    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"

    text_model_name: str = "openai/clip-vit-large-patch14-336"
    text_checkpoint_path: str = "/home/ly/myproject/lw/text_pretrained/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth"
    freeze_text_encoder: bool = True
    text_max_len: int = 300

    lambda_fusion: float = 1.0
    lambda_ts: float = 0.05
    lambda_adv: float = 0.01
    lambda_rec: float = 0.1
    adv_grl_lambda: float = 1.0
    lambda_score: float = 1.0

    epochs: int = 80
    batch_size: int = 16
    batch_size_eval: int = 64
    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    mixed_precision: bool = True
    label_smoothing: float = 0.1
    clip_grad: float = 100.0

    seed: int = 42
    verbose: bool = True
    gpu_ids: tuple = (1,)
    normalize_features: bool = True
    eval_every_n_epoch: int = 1
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    model_path: str = "./cvgtext_rawfusion_runs"


def main():
    config = Configuration()
    train_cities = list(config.city_train)
    eval_cities = list(config.city_eval)
    run_dir = os.path.join(config.model_path, "-".join(train_cities), config.model, time.strftime("%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(run_dir, "log.txt"))

    setup_system(seed=config.seed, cudnn_benchmark=config.cudnn_benchmark, cudnn_deterministic=config.cudnn_deterministic)

    print("Mode: raw_backbone_text_fusion_no_disentangle")
    print(f"Train cities: {train_cities}")
    print(f"Eval cities: {eval_cities}")
    print(f"lambda_score: {config.lambda_score}")

    ckpt = config.text_checkpoint_path if len(config.text_checkpoint_path) > 0 else None
    model = RawFusionGeoModel(
        model_name=config.model,
        pretrained=True,
        img_size=config.img_size,
        pretrained_path=config.pretrained_weight_path,
        text_max_len=config.text_max_len,
        text_model_name=config.text_model_name,
        text_checkpoint_path=ckpt,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    if len(config.text_checkpoint_path) > 0:
        model.text_encoder.inspect_checkpoint(config.text_checkpoint_path)

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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    eval_sets = {}
    for ec in eval_cities:
        ref_eval = CVGTextPhotosEval(config.data_root, ec, split="test", img_type="reference", transforms=sat_tf_val)
        qry_eval = CVGTextPhotosEval(config.data_root, ec, split="test", img_type="query", transforms=grd_tf_val)
        eval_sets[ec] = {
            "ref_loader": DataLoader(ref_eval, batch_size=config.batch_size_eval, shuffle=False, num_workers=config.num_workers, pin_memory=True),
            "qry_loader": DataLoader(qry_eval, batch_size=config.batch_size_eval, shuffle=False, num_workers=config.num_workers, pin_memory=True),
        }
        print(f"Eval [{ec}] ref={len(ref_eval)} qry={len(qry_eval)}")

    ce = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    retrieval_loss_fn = InfoNCE(loss_function=ce, device=config.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=len(train_loader) * config.epochs,
        num_warmup_steps=len(train_loader) * config.warmup_epochs,
    )
    scaler = GradScaler(init_scale=2.0 ** 10) if config.mixed_precision else None

    best_final = -1.0
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'-' * 30}[Epoch: {epoch}]{'-' * 30}")
        losses = train_rawfusion(config, model, train_loader, retrieval_loss_fn, optimizer, scheduler=scheduler, scaler=scaler)
        print(
            "Epoch {} | total={:.4f} retr={:.4f} fuse={:.4f} ts={:.4f} adv={:.4f} rec={:.4f} lr={:.6f}".format(
                epoch,
                losses["loss_total"],
                losses["loss_retr"],
                losses["loss_fusion"],
                losses["loss_ts"],
                losses["loss_adv"],
                losses["loss_rec"],
                optimizer.param_groups[0]["lr"],
            )
        )

        r_img_mean = r_fused_mean = r_final_mean = img_fused_cos_mean = None
        if epoch % config.eval_every_n_epoch == 0 or epoch == config.epochs:
            img_vals, fused_vals, final_vals, img_fused_cos_vals = [], [], [], []
            print("\n------------------------------[Evaluate Raw Fusion]------------------------------")
            for ec in eval_cities:
                r_img, r_fused, r_final, img_fused_cos = evaluate_rawfusion(
                    config,
                    model,
                    eval_sets[ec]["qry_loader"],
                    eval_sets[ec]["ref_loader"],
                    ec,
                )
                img_vals.append(r_img)
                fused_vals.append(r_fused)
                final_vals.append(r_final)
                img_fused_cos_vals.append(img_fused_cos)
            r_img_mean = float(sum(img_vals) / len(img_vals))
            r_fused_mean = float(sum(fused_vals) / len(fused_vals))
            r_final_mean = float(sum(final_vals) / len(final_vals))
            img_fused_cos_mean = float(sum(img_fused_cos_vals) / len(img_fused_cos_vals))
            print(f"R@1 mean raw image-only: {r_img_mean:.4f}")
            print(f"R@1 mean raw fusion-only: {r_fused_mean:.4f}")
            print(f"R@1 mean raw score-fusion: {r_final_mean:.4f}")
            print(f"Mean raw image-fused cosine: {img_fused_cos_mean:.6f}")

            if r_final_mean > best_final:
                best_final = r_final_mean
                save_path = os.path.join(run_dir, f"weights_best_scorefusion_e{epoch}_{r_final_mean:.4f}.pth")
                save_model(model, config, save_path)
                print("Saved best score-fusion:", save_path)

        history.append({
            "epoch": epoch,
            **losses,
            "r1_img": r_img_mean,
            "r1_fused": r_fused_mean,
            "r1_final": r_final_mean,
            "img_fused_cos": img_fused_cos_mean,
            "lr": optimizer.param_groups[0]["lr"],
        })
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    final_path = os.path.join(run_dir, "weights_end.pth")
    save_model(model, config, final_path)
    print("Saved final:", final_path)
    print("Saved outputs:", run_dir)


if __name__ == "__main__":
    main()
