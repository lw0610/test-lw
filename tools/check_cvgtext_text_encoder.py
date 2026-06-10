import argparse
import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sample4geo.models.cvgtext_text_encoder import CVGTextCLIPTextEncoder


def find_first_checkpoint(path: str):
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    candidates = []
    for n in os.listdir(path):
        p = os.path.join(path, n)
        if os.path.isfile(p) and (n.endswith(".pt") or n.endswith(".pth") or n.endswith(".bin") or n.endswith(".ckpt")):
            candidates.append(p)
    candidates.sort()
    return candidates[0] if len(candidates) > 0 else None


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"], True
    if isinstance(ckpt_obj, dict):
        return ckpt_obj, False
    raise ValueError("Unsupported checkpoint format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--text_model_name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--max_text_len", type=int, default=300)
    args = parser.parse_args()

    ckpt_path = find_first_checkpoint(args.checkpoint_path)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint file found in: {args.checkpoint_path}")

    print(f"[check] checkpoint path: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict, wrapped = extract_state_dict(ckpt_obj)

    print(f"[check] wrapped state_dict: {wrapped}")
    if isinstance(ckpt_obj, dict):
        print("[check] top-level keys:", list(ckpt_obj.keys())[:100])

    patterns = ["text", "token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection"]
    pos_shape = None
    for k, v in state_dict.items():
        lk = k.lower()
        if any(p in lk for p in patterns):
            shp = tuple(v.shape) if torch.is_tensor(v) else "-"
            print(f"[check] {k}: {shp}")
            if "positional_embedding" in lk and torch.is_tensor(v):
                pos_shape = tuple(v.shape)

    if pos_shape is not None:
        print(f"[check] detected positional_embedding shape: {pos_shape}")
        if len(pos_shape) == 2 and pos_shape[0] == 300:
            print("[check] positional length is already 300 (no extra interpolation needed)")
        elif len(pos_shape) == 2 and pos_shape[0] == 77:
            print("[check] positional length is 77 (will perform EPE interpolation to 300)")

    enc = CVGTextCLIPTextEncoder(
        model_name=args.text_model_name,
        max_text_len=args.max_text_len,
        checkpoint_path=ckpt_path,
        freeze_text_encoder=True,
        strict_load=False,
    )

    texts = [
        "A street view near high-rise buildings and an intersection.",
        "A suburban area with trees and low residential houses.",
    ]

    with torch.no_grad():
        tokens = enc.tokenize(texts)
        feat = enc(tokens)

    print(f"[check] input_ids shape: {tuple(tokens['input_ids'].shape)}")
    print(f"[check] text feature shape: {tuple(feat.shape)}")
    norms = torch.norm(feat, dim=-1)
    print(f"[check] text feature norms: {norms.tolist()}")
    print(f"[check] output dim (for fusion): {feat.shape[-1]}")


if __name__ == "__main__":
    main()
