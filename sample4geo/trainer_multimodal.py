import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

from sample4geo.utils import AverageMeter
from sample4geo.losses.orth_loss import orth_loss


def train_multimodal(config,
                     model,
                     dataloader,
                     retrieval_loss_fn,
                     optimizer,
                     scheduler=None,
                     scaler=None):
    model.train()

    meter_total = AverageMeter()
    meter_retr = AverageMeter()
    meter_fusion = AverageMeter()
    meter_ts = AverageMeter()
    meter_adv = AverageMeter()
    meter_rec = AverageMeter()
    meter_orth = AverageMeter()

    meter_gate_mean = AverageMeter()
    meter_gate_std = AverageMeter()
    meter_disc_acc = AverageMeter()

    time.sleep(0.1)
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

        if scaler:
            with autocast():
                out = model(query, reference, texts=tokenized, adv_grl_lambda=config.adv_grl_lambda)
                sem_q = out["sem_q"]
                sem_g = out["sem_g"]
                sty_q = out["sty_q"]
                sty_g = out["sty_g"]
                text_emb = out["text_emb"]
                fused_q = out["fused_q"]
                alpha = out["fusion_gate"]
                sem_q_ebr = out["sem_q_ebr"]
                text_emb_ebr = out["text_emb_ebr"]
                disc_logits = out["disc_logits"]

                logit_scale = model.module.logit_scale.exp() if hasattr(model, "module") else model.logit_scale.exp()
                loss_retr = retrieval_loss_fn(sem_q, sem_g, logit_scale)

                loss_fusion = torch.zeros((), device=query.device)
                loss_ts = torch.zeros((), device=query.device)
                loss_adv = torch.zeros((), device=query.device)
                loss_rec = torch.zeros((), device=query.device)
                loss_orth = torch.zeros((), device=query.device)

                if config.train_mode == "text_fusion" and text_emb is not None:
                    loss_fusion = retrieval_loss_fn(fused_q, sem_g, logit_scale)
                    loss_ts = retrieval_loss_fn(text_emb, sem_g.detach(), logit_scale)

                    rec_img = 1.0 - (sem_q_ebr * sem_q).sum(dim=-1).mean()
                    rec_txt = 1.0 - (text_emb_ebr * text_emb).sum(dim=-1).mean()
                    loss_rec = 0.5 * (rec_img + rec_txt)

                    if disc_logits is not None:
                        bsz = sem_q.shape[0]
                        labels = torch.cat([
                            torch.zeros(bsz, dtype=torch.long, device=query.device),
                            torch.ones(bsz, dtype=torch.long, device=query.device),
                        ], dim=0)
                        loss_adv = F.cross_entropy(disc_logits, labels)

                if config.train_mode == "baseline_plus_text_plus_orth":
                    loss_orth = orth_loss(sem_q, sty_q) + orth_loss(sem_g, sty_g)

                loss = (
                    loss_retr
                    + config.lambda_fusion * loss_fusion
                    + config.lambda_ts * loss_ts
                    + config.lambda_adv * loss_adv
                    + config.lambda_rec * loss_rec
                    + config.lambda_orth * loss_orth
                )

            scaler.scale(loss).backward()

            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        else:
            out = model(query, reference, texts=tokenized, adv_grl_lambda=config.adv_grl_lambda)
            sem_q = out["sem_q"]
            sem_g = out["sem_g"]
            sty_q = out["sty_q"]
            sty_g = out["sty_g"]
            text_emb = out["text_emb"]
            fused_q = out["fused_q"]
            alpha = out["fusion_gate"]
            sem_q_ebr = out["sem_q_ebr"]
            text_emb_ebr = out["text_emb_ebr"]
            disc_logits = out["disc_logits"]

            logit_scale = model.module.logit_scale.exp() if hasattr(model, "module") else model.logit_scale.exp()
            loss_retr = retrieval_loss_fn(sem_q, sem_g, logit_scale)

            loss_fusion = torch.zeros((), device=query.device)
            loss_ts = torch.zeros((), device=query.device)
            loss_adv = torch.zeros((), device=query.device)
            loss_rec = torch.zeros((), device=query.device)
            loss_orth = torch.zeros((), device=query.device)

            if config.train_mode == "text_fusion" and text_emb is not None:
                loss_fusion = retrieval_loss_fn(fused_q, sem_g, logit_scale)
                loss_ts = retrieval_loss_fn(text_emb, sem_g.detach(), logit_scale)

                rec_img = 1.0 - (sem_q_ebr * sem_q).sum(dim=-1).mean()
                rec_txt = 1.0 - (text_emb_ebr * text_emb).sum(dim=-1).mean()
                loss_rec = 0.5 * (rec_img + rec_txt)

                if disc_logits is not None:
                    bsz = sem_q.shape[0]
                    labels = torch.cat([
                        torch.zeros(bsz, dtype=torch.long, device=query.device),
                        torch.ones(bsz, dtype=torch.long, device=query.device),
                    ], dim=0)
                    loss_adv = F.cross_entropy(disc_logits, labels)

            if config.train_mode == "baseline_plus_text_plus_orth":
                loss_orth = orth_loss(sem_q, sty_q) + orth_loss(sem_g, sty_g)

            loss = (
                loss_retr
                + config.lambda_fusion * loss_fusion
                + config.lambda_ts * loss_ts
                + config.lambda_adv * loss_adv
                + config.lambda_rec * loss_rec
                + config.lambda_orth * loss_orth
            )
            loss.backward()

            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if config.train_mode == "text_fusion" and alpha is not None:
            with torch.no_grad():
                meter_gate_mean.update(alpha.mean().item())
                meter_gate_std.update(alpha.std().item())

                if disc_logits is not None:
                    bsz = sem_q.shape[0]
                    labels = torch.cat([
                        torch.zeros(bsz, dtype=torch.long, device=query.device),
                        torch.ones(bsz, dtype=torch.long, device=query.device),
                    ], dim=0)
                    pred = disc_logits.argmax(dim=-1)
                    acc = (pred == labels).float().mean().item()
                    meter_disc_acc.update(acc)

        if scheduler is not None and config.scheduler in ["polynomial", "cosine", "constant"]:
            scheduler.step()

        meter_total.update(loss.item())
        meter_retr.update(loss_retr.item())
        meter_fusion.update(loss_fusion.item() if torch.is_tensor(loss_fusion) else float(loss_fusion))
        meter_ts.update(loss_ts.item() if torch.is_tensor(loss_ts) else float(loss_ts))
        meter_adv.update(loss_adv.item() if torch.is_tensor(loss_adv) else float(loss_adv))
        meter_rec.update(loss_rec.item() if torch.is_tensor(loss_rec) else float(loss_rec))
        meter_orth.update(loss_orth.item() if torch.is_tensor(loss_orth) else float(loss_orth))

        if config.verbose:
            bar.set_postfix(ordered_dict={
                "total": f"{meter_total.avg:.4f}",
                "retr": f"{meter_retr.avg:.4f}",
                "fuse": f"{meter_fusion.avg:.4f}",
                "ts": f"{meter_ts.avg:.4f}",
                "adv": f"{meter_adv.avg:.4f}",
                "rec": f"{meter_rec.avg:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

    if config.verbose:
        bar.close()

    if config.train_mode == "text_fusion" and meter_gate_mean.count > 0:
        print(
            "[Epoch Gate Stats] "
            f"mean={meter_gate_mean.avg:.4f}, "
            f"std={meter_gate_std.avg:.4f}, "
            f"disc_acc={meter_disc_acc.avg:.4f}"
        )

    return {
        "loss_total": meter_total.avg,
        "loss_retr": meter_retr.avg,
        "loss_txt": meter_fusion.avg,
        "loss_ts": meter_ts.avg,
        "loss_adv": meter_adv.avg,
        "loss_rec": meter_rec.avg,
        "loss_orth": meter_orth.avg,
    }
