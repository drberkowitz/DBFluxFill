"""
download_models.py
Downloads FLUX.1 Fill Dev model components from HuggingFace.
Called by the installer during first-time setup.
Not needed at runtime.

Usage:
    python download_models.py --token <HF_TOKEN> --output <models_folder> --variant <bf16|fp8|gguf>

Output structure (all variants):
    models/
        transformer/
        vae/
        text_encoder/
        text_encoder_2/
        tokenizer/
        tokenizer_2/
        scheduler/
"""

import argparse
import os
import sys
import traceback
from huggingface_hub import snapshot_download, hf_hub_download


BFL_REPO   = "black-forest-labs/FLUX.1-Fill-dev"
FP8_REPO   = "AlekseyCalvin/FluxFillDev_fp8_Diffusers"
GGUF_REPO  = "YarvixPA/FLUX.1-Fill-dev-GGUF"
GGUF_FILE  = "flux1-fill-dev-Q5_1.gguf"

SHARED_ALLOW_PATTERNS = [
    "vae/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "scheduler/*",
    "model_index.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download FLUX Fill Dev models")
    parser.add_argument("--token",   required=True, help="HuggingFace access token")
    parser.add_argument("--output",  required=True, help="Folder to save models into")
    parser.add_argument("--variant", required=True, choices=["bf16", "fp8", "gguf"],
                        help="Model variant to download")
    return parser.parse_args()


def download_shared_components(token, output_dir):
    print("INFO: Downloading shared components (vae, encoders, tokenizers, scheduler).")
    print("INFO: Download size approximately 10.4 GB.")
    snapshot_download(
        repo_id=BFL_REPO,
        local_dir=output_dir,
        token=token,
        allow_patterns=SHARED_ALLOW_PATTERNS,
    )
    print("INFO: Shared components downloaded.")


def download_bf16(token, output_dir):
    print("INFO: Variant: BF16 (official BFL release).")
    print("INFO: Downloading full diffusers pipeline, approximately 34.3 GB.")
    snapshot_download(
        repo_id=BFL_REPO,
        local_dir=output_dir,
        token=token,
        ignore_patterns=[
            "flux1-fill-dev.safetensors",
            "*.gguf",
            ".cache/*",
        ],
    )
    print("INFO: BF16 download complete.")


def download_fp8(token, output_dir):
    print("INFO: Variant: FP8 (community release by AlekseyCalvin).")
    print("INFO: Downloading full FP8 diffusers pipeline, approximately 22.3 GB.")
    snapshot_download(
        repo_id=FP8_REPO,
        local_dir=output_dir,
        token=token,
        ignore_patterns=[
            ".cache/*",
        ],
    )
    print("INFO: FP8 download complete.")

    # Overwrite transformer/config.json with the official BFL version
    print("INFO: Patching transformer/config.json with official BFL version.")

    try:
        hf_hub_download(
            repo_id=BFL_REPO,
            filename="transformer/config.json",
            local_dir=output_dir,
            token=token,
        )
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err or "gated" in err.lower() or "access" in err.lower():
            print("ERROR: Access denied to the BFL FluxFill repo.")
            print("ERROR: Ensure your token has read access.")
        else:
            print("ERROR: BFL FluxFill config.json download failed: {}".format(e))
        sys.exit(1)
    print("INFO: transformer/config.json patched.")


def download_gguf(token, output_dir):
    print("INFO: Variant: GGUF Q5_1 (community release by YarvixPA).")

    # Shared components first
    download_shared_components(token, output_dir)

    # GGUF transformer
    print("INFO: Downloading GGUF transformer ({}), approximately 12.7 GB.".format(GGUF_FILE))
    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=transformer_dir,
            token=token,
        )
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err or "gated" in err.lower() or "access" in err.lower():
            print("ERROR: Access denied to the GGUF repo.")
            print("ERROR: Ensure your token has read access.")
        else:
            print("ERROR: GGUF transformer download failed: {}".format(e))
        sys.exit(1)

    print("INFO: GGUF transformer downloaded.")
    print("INFO: Downloading official BFL transformer/config.json version.")
    try:
        hf_hub_download(
            repo_id=BFL_REPO,
            filename="transformer/config.json",
            local_dir=output_dir,
            token=token,
        )
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err or "gated" in err.lower() or "access" in err.lower():
            print("ERROR: Access denied to the BFL FluxFill repo.")
            print("ERROR: Ensure your token has read access.")
        else:
            print("ERROR: BFL FluxFill config.json download failed: {}".format(e))
        sys.exit(1)
    print("INFO: transformer/config.json patched.")


def download(token, output_dir, variant):
    os.makedirs(output_dir, exist_ok=True)
    print("INFO: Output directory: {}".format(output_dir))

    try:
        if variant == "bf16":
            download_bf16(token, output_dir)
        elif variant == "fp8":
            download_fp8(token, output_dir)
        elif variant == "gguf":
            download_gguf(token, output_dir)
    except SystemExit:
        raise
    except Exception as e:
        print("ERROR: Download failed: {}".format(e))
        traceback.print_exc()
        sys.exit(1)

    print("INFO: All components saved to: {}".format(output_dir))


if __name__ == "__main__":
    args = parse_args()
    download(args.token, args.output, args.variant)
    print("INFO: download_models.py finished.")
    sys.exit(0)