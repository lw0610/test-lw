import os
import time
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

from torch.utils.data import DataLoader
from sample4geo.dataset.cvusa import CVUSADatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from sample4geo.trainer import predict
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'

    # Override model image size
    img_size: int = 384

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True

    # Dataset
    data_folder = "/home/ly/CVUSA/CVPR_subset"

    # Local timm pretrained weights (.safetensors)
    pretrained_weight_path: str = "/home/ly/myproject/lw/Sample4Geo-main/pretrained/model.safetensors"

    # Checkpoint to evaluate
    checkpoint_start = "/home/ly/myproject/lw/Sample4Geo-main/cvusa/convnext_base.fb_in22k_ft_in1k_384/223334/weights_e40_96.2629.pth"

    # Visualization output
    save_root: str = './eval_results_cvusa'
    retrieval_k: int = 5
    vis_samples: int = 50
    save_wrong_max: int = 100

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_topk_and_wrong_cases(config, model, reference_dataloader, query_dataloader, reference_dataset, query_dataset, save_dir):
    print("\nGenerating retrieval visualizations...")

    reference_features, reference_labels = predict(config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    sim = query_features @ reference_features.T
    _, topk_ids = torch.topk(sim, k=config.retrieval_k, dim=1)

    idx2ref = {int(i): os.path.join(config.data_folder, p) for i, p in zip(reference_dataset.label, reference_dataset.images)}
    idx2qry = {int(i): os.path.join(config.data_folder, p) for i, p in zip(query_dataset.label, query_dataset.images)}

    topk_dir = os.path.join(save_dir, 'retrieval_topk')
    wrong_dir = os.path.join(save_dir, 'wrong_cases_top1')
    os.makedirs(topk_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    n_vis = min(config.vis_samples, len(query_labels))
    wrong_saved = 0

    for i in range(len(query_labels)):
        qid = int(query_labels[i].item())

        pred_ids = [int(reference_labels[pos].item()) for pos in topk_ids[i].cpu().tolist()]
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


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':
    run_dir = os.path.join(config.save_root, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

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

    # load checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                    img_size_ground,
                                                                    mean=mean,
                                                                    std=std)

    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val)

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder,
                                          split="test",
                                          img_type="query",
                                          transforms=ground_transforms_val)

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#

    print("\n{}[{}]{}".format(30*"-", "CVUSA", 30*"-"))

    _ = evaluate(config=config,
                 model=model,
                 reference_dataloader=reference_dataloader_test,
                 query_dataloader=query_dataloader_test,
                 ranks=[1, 5, 10],
                 step_size=1000,
                 cleanup=True)

    save_topk_and_wrong_cases(config,
                              model,
                              reference_dataloader_test,
                              query_dataloader_test,
                              reference_dataset_test,
                              query_dataset_test,
                              run_dir)

    print("\nSaved outputs:")
    print(f"- {run_dir}")
    print("  - retrieval_topk/*.png")
    print("  - wrong_cases_top1/*.png (max 100)")
