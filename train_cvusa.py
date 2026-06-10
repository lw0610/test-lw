import os
import time
import shutil
import sys
import random
import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train, predict
from sample4geo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384

    # Local timm pretrained weights (.safetensors)
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"
    
    # Training
    mixed_precision: bool = True
    seed = 42
    epochs: int = 40
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (1,)

    # Subset mode (for quick smoke-run)
    subset_mode: bool = False
    train_subset_size: int = 2000
    val_subset_size: int = 1000

    # Similarity Sampling
    custom_sampling: bool = False
    gps_sample: bool = False
    sim_sample: bool = False
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_dict_path: str = "/home/ly/myproject/lw/Sample4Geo-main/gps_dict.pkl"
 
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 2
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001  #作者是0.001
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 0.0001
    
    # Dataset
    data_folder = "/home/ly/CVUSA/CVPR_subset"
    
    # Augment Images
    prob_rotate: float = 0.5
    prob_flip: float = 0.5
    
    # Savepath for model checkpoints
    model_path: str = "./cvusa"

    # Visualization
    retrieval_k: int = 5
    retrieval_vis_samples: int = 8
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False


def build_subset_indices(n_total: int, n_subset: int, seed: int):
    n_subset = min(n_total, n_subset)
    idx = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[:n_subset]


def save_loss_curve(history, save_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    # Loss curve only
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss", color="tab:blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=180)
    plt.close()

    # Recall curve only
    if "r1_test" in df.columns:
        recall_df = df[df["r1_test"].notna()]
        if len(recall_df) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(recall_df["epoch"], recall_df["r1_test"], marker="s", label="Recall@1", color="tab:orange")
            plt.xlabel("Epoch")
            plt.ylabel("Recall@1 (%)")
            plt.title("Validation Recall Curve")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "recall_curve.png"), dpi=180)
            plt.close()


def save_topk_retrieval_figure(config, model, reference_loader, query_loader, reference_dataset_base, query_dataset_base, save_dir):
    print("\nGenerating top-k retrieval figures...")
    reference_features, reference_labels = predict(config, model, reference_loader)
    query_features, query_labels = predict(config, model, query_loader)

    sim = query_features @ reference_features.T
    _, topk_ids = torch.topk(sim, k=config.retrieval_k, dim=1)

    idx2ref = {int(i): os.path.join(config.data_folder, p) for i, p in zip(reference_dataset_base.label, reference_dataset_base.images)}
    idx2qry = {int(i): os.path.join(config.data_folder, p) for i, p in zip(query_dataset_base.label, query_dataset_base.images)}

    retrieval_dir = os.path.join(save_dir, "retrieval_topk")
    os.makedirs(retrieval_dir, exist_ok=True)

    n_vis = min(config.retrieval_vis_samples, len(query_labels))
    for i in range(n_vis):
        qid = int(query_labels[i].item())
        qimg = plt.imread(idx2qry[qid])

        # Query + GT + Top-k predictions
        cols = 2 + config.retrieval_k
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))

        axes[0].imshow(qimg)
        axes[0].set_title(f"Query\nID={qid}")
        axes[0].axis("off")

        gt_img = plt.imread(idx2ref[qid])
        axes[1].imshow(gt_img)
        axes[1].set_title(f"GT Ref\nID={qid}")
        axes[1].axis("off")

        for j, ref_pos in enumerate(topk_ids[i].cpu().tolist(), start=2):
            rank = j - 1
            rid = int(reference_labels[ref_pos].item())
            rimg = plt.imread(idx2ref[rid])
            axes[j].imshow(rimg)
            axes[j].set_title(f"Top-{rank}\nID={rid}\n{'HIT' if rid == qid else 'MISS'}")
            axes[j].axis("off")

            if rid == qid:
                for spine in axes[j].spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig(os.path.join(retrieval_dir, f"query_{i:03d}_id_{qid}.png"), dpi=160)
        plt.close(fig)


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    print("\nModel: {}".format(config.model))

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      pretrained_path=config.pretrained_weight_path)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    train_dataset = CVUSADatasetTrain(data_folder=config.data_folder,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )

    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )

    reference_dataset_test_base = CVUSADatasetEval(data_folder=config.data_folder,
                                                   split="test",
                                                   img_type="reference",
                                                   transforms=sat_transforms_val,
                                                   )

    query_dataset_test_base = CVUSADatasetEval(data_folder=config.data_folder,
                                               split="test",
                                               img_type="query",
                                               transforms=ground_transforms_val,
                                               )

    if config.subset_mode:
        train_idx = build_subset_indices(len(train_dataset), config.train_subset_size, config.seed)
        val_idx = build_subset_indices(len(reference_dataset_test_base), config.val_subset_size, config.seed)

        train_dataset = Subset(train_dataset, train_idx)
        reference_dataset_test = Subset(reference_dataset_test_base, val_idx)
        query_dataset_test = Subset(query_dataset_test_base, val_idx)

        print(f"Subset mode ON | Train: {len(train_dataset)} | Val: {len(reference_dataset_test)}")
    else:
        reference_dataset_test = reference_dataset_test_base
        query_dataset_test = query_dataset_test_base

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    if config.sim_sample:

        query_dataset_train = CVUSADatasetEval(data_folder=config.data_folder,
                                               split="train",
                                               img_type="query",
                                               transforms=ground_transforms_val,
                                               )

        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)

        reference_dataset_train = CVUSADatasetEval(data_folder=config.data_folder,
                                                   split="train",
                                                   img_type="reference",
                                                   transforms=sat_transforms_val,
                                                   )

        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)

        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train,
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)

    if config.custom_sampling and hasattr(train_dataloader.dataset, "shuffle"):
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)

    best_score = 0
    history = []

    for epoch in range(1, config.epochs+1):

        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        r1_test = None
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)

            if r1_test > best_score:
                best_score = r1_test
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "r1_test": None if r1_test is None else float(r1_test),
            "lr": float(optimizer.param_groups[0]['lr'])
        })

        if config.custom_sampling and hasattr(train_dataloader.dataset, "shuffle"):
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    save_loss_curve(history, model_path)

    save_topk_retrieval_figure(config,
                               model,
                               reference_dataloader_test,
                               query_dataloader_test,
                               reference_dataset_test_base,
                               query_dataset_test_base,
                               model_path)

    print("\nSaved outputs:")
    print("- log.txt")
    print("- history.csv")
    print("- loss_curve.png")
    print("- recall_curve.png")
    print("- retrieval_topk/*.png (Query + GT + Top-k)")
