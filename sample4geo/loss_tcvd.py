# sample4geo/loss_tcvd.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_info_nce(features1, features2, logit_scale, label_smoothing=0.0):
    """
    对称 InfoNCE 损失。

    features1: [B, D]
    features2: [B, D]

    默认第 i 个 features1 和第 i 个 features2 是正样本。
    例如：
        c_q[i] 和 c_s[i] 是同一地点的 ground/satellite 正样本；
        c_q[i] 和 text_emb[i] 是同一地点的图像/文本正样本。

    logit_scale:
        通常是 model.logit_scale.exp()
    """

    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)

    if torch.is_tensor(logit_scale) and logit_scale.numel() > 1:
            logit_scale = logit_scale.mean()
            
    logits = logit_scale * features1 @ features2.T

    labels = torch.arange(
        logits.size(0),
        dtype=torch.long,
        device=logits.device,
    )

    loss_12 = F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )

    loss_21 = F.cross_entropy(
        logits.T,
        labels,
        label_smoothing=label_smoothing,
    )

    loss = 0.5 * (loss_12 + loss_21)

    return loss


def content_viewpoint_decorrelation_loss(content, viewpoint, eps=1e-6):
    """
    content-viewpoint 独立性约束。

    这里不用简单的 dot product 正交，而是使用 batch 维度上的
    cross-covariance decorrelation。

    目的：
        让 content 分支和 viewpoint 分支在统计上尽量不相关，
        避免 content 里面混入太多视角信息，
        也避免 viewpoint 里面混入太多地点语义信息。

    输入：
        content:   [B, C]
        viewpoint: [B, V]

    输出：
        一个标量 loss。
    """

    # batch 太小时无法稳定估计协方差，直接返回 0
    if content.size(0) <= 1:
        return content.new_tensor(0.0)

    # 去均值
    content = content - content.mean(dim=0, keepdim=True)
    viewpoint = viewpoint - viewpoint.mean(dim=0, keepdim=True)

    # 标准化，避免某些维度尺度过大主导 loss
    content = content / (content.std(dim=0, keepdim=True) + eps)
    viewpoint = viewpoint / (viewpoint.std(dim=0, keepdim=True) + eps)

    # cross-covariance: [C, V]
    cov = content.T @ viewpoint / (content.size(0) - 1)

    # 希望所有 content 维度和 viewpoint 维度之间的相关性都接近 0
    loss = cov.pow(2).mean()

    return loss


class TCVDLoss(nn.Module):
    """
    Text-Guided Content-Viewpoint Disentanglement Loss.

    总损失：

        L = L_loc
            + lambda_txt * L_txt
            + lambda_iic * L_iic
            + lambda_rec * L_rec

    其中：

    1. L_loc:
        跨视角定位损失。
        使用 query content 和 satellite content 做 InfoNCE。

    2. L_txt:
        文本指导损失。
        使用文本特征分别对齐 query content 和 satellite content。
        文本只指导 content，不指导 viewpoint。

    3. L_iic:
        intra-view independence constraint。
        约束 content 和 viewpoint 在同一视角内部尽量独立。

    4. L_rec:
        inter-view reconstruction constraint。
        用 c_s + v_q 重建 z_q，
        用 c_q + v_s 重建 z_s，
        防止 content/viewpoint 分支坍塌。
    """

    def __init__(
        self,
        lambda_txt=0.05,
        lambda_iic=0.05,
        lambda_rec=0.1,
        label_smoothing=0.1,
        use_text_loss=True,
        use_iic_loss=True,
        use_rec_loss=True,
    ):
        super().__init__()

        self.lambda_txt = lambda_txt
        self.lambda_iic = lambda_iic
        self.lambda_rec = lambda_rec

        self.label_smoothing = label_smoothing

        self.use_text_loss = use_text_loss
        self.use_iic_loss = use_iic_loss
        self.use_rec_loss = use_rec_loss

    def forward(self, out):
        """
        out 是 model_tcvd.py 中 TextGuidedCVDModel forward 返回的字典。

        需要包含：

            out["c_q"]       : query content，归一化后，用于检索
            out["c_s"]       : satellite content，归一化后，用于检索
            out["c_q_raw"]   : query content，未归一化，用于解耦/重建
            out["c_s_raw"]   : satellite content，未归一化，用于解耦/重建
            out["v_q"]       : query viewpoint
            out["v_s"]       : satellite viewpoint
            out["z_q"]       : query backbone raw feature
            out["z_s"]       : satellite backbone raw feature
            out["rec_q"]     : c_s + v_q 重建得到的 query feature
            out["rec_s"]     : c_q + v_s 重建得到的 satellite feature
            out["text_emb"]  : 文本特征，可选
            out["logit_scale"]: 温度缩放参数
        """

        logit_scale = out["logit_scale"]

        # ============================================================
        # 1. 跨视角 content 检索损失 L_loc
        # ============================================================
        loss_loc = symmetric_info_nce(
            out["c_q"],
            out["c_s"],
            logit_scale=logit_scale,
            label_smoothing=self.label_smoothing,
        )

        # ============================================================
        # 2. 文本指导 content 损失 L_txt
        # ============================================================
        loss_txt = loss_loc.new_tensor(0.0)

        if self.use_text_loss and out.get("text_emb", None) is not None:
            text_emb = out["text_emb"]

            # 文本指导 query content
            loss_txt_q = symmetric_info_nce(
                out["c_q"],
                text_emb,
                logit_scale=logit_scale,
                label_smoothing=0.0,
            )

            # 文本指导 satellite content
            loss_txt_s = symmetric_info_nce(
                out["c_s"],
                text_emb,
                logit_scale=logit_scale,
                label_smoothing=0.0,
            )

            loss_txt = 0.5 * (loss_txt_q + loss_txt_s)

        # ============================================================
        # 3. content-viewpoint 独立性约束 L_iic
        # ============================================================
        loss_iic = loss_loc.new_tensor(0.0)

        if self.use_iic_loss:
            loss_iic_q = content_viewpoint_decorrelation_loss(
                out["c_q_raw"],
                out["v_q"],
            )

            loss_iic_s = content_viewpoint_decorrelation_loss(
                out["c_s_raw"],
                out["v_s"],
            )

            loss_iic = 0.5 * (loss_iic_q + loss_iic_s)

        # ============================================================
        # 4. 跨视角特征重建损失 L_rec
        # ============================================================
        loss_rec = loss_loc.new_tensor(0.0)

        if self.use_rec_loss:
            # detach target，避免重建损失过度拉动 backbone 原始特征空间
            loss_rec_q = F.mse_loss(
                out["rec_q"],
                out["z_q"].detach(),
            )

            loss_rec_s = F.mse_loss(
                out["rec_s"],
                out["z_s"].detach(),
            )

            loss_rec = 0.5 * (loss_rec_q + loss_rec_s)

        # ============================================================
        # 5. 总损失
        # ============================================================
        loss_total = (
            loss_loc
            + self.lambda_txt * loss_txt
            + self.lambda_iic * loss_iic
            + self.lambda_rec * loss_rec
        )

        loss_dict = {
            "loss_total": loss_total.detach(),
            "loss_loc": loss_loc.detach(),
            "loss_txt": loss_txt.detach(),
            "loss_iic": loss_iic.detach(),
            "loss_rec": loss_rec.detach(),
        }

        return loss_total, loss_dict