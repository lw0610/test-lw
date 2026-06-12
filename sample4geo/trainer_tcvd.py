# sample4geo/trainer_tcvd.py

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast

from sample4geo.utils import AverageMeter


def _get_core_model(model):
    """
    兼容 DataParallel。
    如果 model 被 torch.nn.DataParallel 包起来，
    真正的模型在 model.module 里面。
    """
    return model.module if hasattr(model, "module") else model


def _use_amp(config, scaler):
    """
    是否启用 AMP 混合精度。
    只有在 CUDA 可用且 scaler 不为 None 时启用。
    """
    if scaler is None:
        return False

    if not torch.cuda.is_available():
        return False

    device = str(getattr(config, "device", ""))
    return device.startswith("cuda")


def _tokenize_texts(config, model, texts):
    """
    把 dataloader 返回的文本 list[str] 转成 token tensor。

    你的 CVGTextPhotosTrain 返回：
        query, reference, text, label

    DataLoader 会把 text 整理成 list[str] 或 tuple[str]。
    这里使用 model.text_encoder.tokenize()，和你原来的 trainer_multimodal.py 保持一致。
    """
    core_model = _get_core_model(model)

    if texts is None:
        return None

    # 如果已经是 tensor，说明外部已经 tokenize 过，直接放到 device
    if torch.is_tensor(texts):
        return texts.to(config.device)

    # 一般情况：texts 是 list[str] 或 tuple[str]
    texts = list(texts)

    tokenized = core_model.text_encoder.tokenize(texts)

    if isinstance(tokenized, dict):
        tokenized = {
            k: v.to(config.device)
            for k, v in tokenized.items()
        }
    else:
        tokenized = tokenized.to(config.device)

    return tokenized


def train_tcvd(
    config,
    model,
    dataloader,
    loss_function,
    optimizer,
    scheduler=None,
    scaler=None,
):
    """
    T-CVD 训练函数。

    dataloader 每个 batch 应该返回：
        query, reference, texts, labels

    对应你的 CVGTextPhotosTrain：
        q_img, r_img, text, label

    训练流程：
        1. query/reference 图像送入 TextGuidedCVDModel
        2. 文本 tokenize 后送入模型
        3. 模型输出 content/viewpoint/text/reconstruction 等中间结果
        4. TCVDLoss 计算：
            loss_loc
            loss_txt
            loss_iic
            loss_rec
        5. 反向传播更新模型
    """

    model.train()

    meter_total = AverageMeter()
    meter_loc = AverageMeter()
    meter_txt = AverageMeter()
    meter_iic = AverageMeter()
    meter_rec = AverageMeter()

    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(
        dataloader,
        total=len(dataloader),
        mininterval=10.0,
    ) if config.verbose else dataloader

    amp_enabled = _use_amp(config, scaler)

    for query, reference, texts, _ in bar:
        query = query.to(config.device, non_blocking=True)
        reference = reference.to(config.device, non_blocking=True)

        tokenized_texts = _tokenize_texts(
            config=config,
            model=model,
            texts=texts,
        )

        if amp_enabled:
            with autocast(enabled=True):
                out = model(
                    query,
                    reference,
                    texts=tokenized_texts,
                )

                loss, loss_dict = loss_function(out)

            scaler.scale(loss).backward()

            if getattr(config, "clip_grad", None):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    config.clip_grad,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        else:
            out = model(
                query,
                reference,
                texts=tokenized_texts,
            )

            loss, loss_dict = loss_function(out)

            loss.backward()

            if getattr(config, "clip_grad", None):
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    config.clip_grad,
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            if getattr(config, "scheduler", None) in ["polynomial", "cosine", "constant"]:
                scheduler.step()

        meter_total.update(loss_dict["loss_total"].item())
        meter_loc.update(loss_dict["loss_loc"].item())
        meter_txt.update(loss_dict["loss_txt"].item())
        meter_iic.update(loss_dict["loss_iic"].item())
        meter_rec.update(loss_dict["loss_rec"].item())

        if config.verbose:
            bar.set_postfix(
                ordered_dict={
                    "total": f"{meter_total.avg:.4f}",
                    "loc": f"{meter_loc.avg:.4f}",
                    "txt": f"{meter_txt.avg:.4f}",
                    "iic": f"{meter_iic.avg:.4f}",
                    "rec": f"{meter_rec.avg:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

    if config.verbose:
        bar.close()

    return {
        "loss_total": meter_total.avg,
        "loss_loc": meter_loc.avg,
        "loss_txt": meter_txt.avg,
        "loss_iic": meter_iic.avg,
        "loss_rec": meter_rec.avg,

        # 下面这些字段是为了兼容你原来 train_cvgtext.py 的日志命名习惯
        "loss_retr": meter_loc.avg,
        "loss_ts": meter_txt.avg,
        "loss_adv": 0.0,
        "loss_orth": meter_iic.avg,
    }


def predict_tcvd(config, model, dataloader):
    """
    T-CVD 推理函数。

    推理阶段不输入文本，只输入单张图像：

        feature = model(img)

    在 TextGuidedCVDModel 里面：
        如果 reference_img is None，
        model(img) 会返回 content feature。

    所以最终检索用的是：

        query_content @ reference_content.T

    不是 fused_q，也不是文本融合分数。
    """

    model.eval()

    time.sleep(0.1)

    bar = tqdm(
        dataloader,
        total=len(dataloader),
        mininterval=10.0,
    ) if config.verbose else dataloader

    img_features_list = []
    ids_list = []

    amp_enabled = torch.cuda.is_available() and str(getattr(config, "device", "")).startswith("cuda")

    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)

            img = img.to(config.device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                img_feature = model(img)

                if getattr(config, "normalize_features", True):
                    img_feature = F.normalize(img_feature, dim=-1)

            img_features_list.append(img_feature.to(torch.float32))

        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(config.device)

    if config.verbose:
        bar.close()

    return img_features, ids_list


# 可选别名：
# 如果你后面某些代码想 from sample4geo.trainer_tcvd import predict，
# 也可以正常使用。
predict = predict_tcvd