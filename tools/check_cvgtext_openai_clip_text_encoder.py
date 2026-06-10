import os
import sys
import argparse
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sample4geo.models.cvgtext_openai_clip_text_encoder import CVGTextOpenAIClipTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/ly/myproject/lw/text_pretrained/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth",
    )
    parser.add_argument("--tokenizer_name", type=str, default="openai/clip-vit-large-patch14-336")
    args = parser.parse_args()

    enc = CVGTextOpenAIClipTextEncoder(
        checkpoint_path=args.checkpoint_path,
        model_name_for_tokenizer=args.tokenizer_name,
        context_length=300,
        vocab_size=49408,
        width=768,
        layers=12,
        heads=12,
        normalize=True,
        freeze_text_encoder=True,
    )

    info = enc.load_info

    print("[check] wrapped_state_dict:", info["wrapped_state_dict"])
    print("[check] checkpoint text keys:", info["checkpoint_text_key_count"])
    print("[check] loaded text keys:", info["loaded_text_key_count"])
    print("[check] missing_keys count:", len(info["missing_keys"]))
    print("[check] missing_keys first20:", info["missing_keys"][:20])
    print("[check] unexpected_keys count:", len(info["unexpected_keys"]))
    print("[check] unexpected_keys first20:", info["unexpected_keys"][:20])

    loaded = set(info["loaded_keys"])

    has_token_embedding = "token_embedding.weight" in loaded
    has_positional = "positional_embedding" in loaded
    has_ln_final_w = "ln_final.weight" in loaded
    has_ln_final_b = "ln_final.bias" in loaded
    has_text_projection = "text_projection" in loaded

    print("[check] token_embedding loaded:", has_token_embedding)
    print("[check] positional_embedding loaded:", has_positional)
    print("[check] ln_final.weight loaded:", has_ln_final_w)
    print("[check] ln_final.bias loaded:", has_ln_final_b)
    print("[check] text_projection loaded:", has_text_projection)

    # transformer.resblocks 12/12 check
    hit_blocks = 0
    for i in range(12):
        prefix = f"transformer.resblocks.{i}."
        any_hit = any(k.startswith(prefix) for k in loaded)
        if any_hit:
            hit_blocks += 1
    print(f"[check] transformer.resblocks hit: {hit_blocks}/12")

    # tokenizer / EOT checks
    bos_id = enc.tokenizer.bos_token_id
    eot_id = enc.tokenizer.eos_token_id
    if bos_id is None:
        bos_id = 49406
    if eot_id is None:
        eot_id = 49407

    print("[check] tokenizer bos_id:", bos_id)
    print("[check] tokenizer eot_id:", eot_id)

    texts = [
        "A busy urban street with high-rise buildings and crossroads.",
        "A suburban neighborhood with trees and detached houses.",
    ]

    tokens = enc.tokenize(texts, max_text_len=300)
    has_eot = []
    for i in range(tokens.shape[0]):
        has_eot.append(bool((tokens[i] == eot_id).any().item()))

    print("[check] tokens shape:", tuple(tokens.shape))
    print("[check] eot exists per sample:", has_eot)

    with torch.no_grad():
        feat = enc.encode_text(tokens)

    print("[check] text feature shape:", tuple(feat.shape))
    print("[check] text feature norms:", torch.norm(feat, dim=-1).cpu().tolist())


if __name__ == "__main__":
    main()
