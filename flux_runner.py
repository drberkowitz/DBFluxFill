"""
flux_runner.py
Standalone inference script for FLUX.1 Fill Dev.
Called as a subprocess by the FluxFillGizmo in Nuke.

Expects models downloaded in diffusers format via download_models.py.
No internet connection required at runtime.

Usage:
    python flux_runner.py \
        --input    <path to input PNG> \
        --mask     <path to mask PNG> \
        --output   <path to write output PNG> \
        --models   <path to models folder (diffusers format)> \
        --steps    <int, default 20> \
        --guidance <float, default 2.5> \
        --seed     <int, -1 for random>

Exit codes:
    0  success
    1  argument error
    2  model load error
    3  inference error
    4  output write error
"""

import argparse
import json
import os
import sys
import random
import traceback
import time

# Block all network calls before any HF imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Live log writer
# ---------------------------------------------------------------------------

_log_path = None

def _log(msg):
    """Print to stdout and append to log file if one is set."""
    print(msg)
    if _log_path:
        try:
            with open(_log_path, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config():
    """Read config.json from the same directory as this script."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print("ERROR: Could not read config.json: {}".format(e), file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1 Fill Dev inference runner")
    parser.add_argument("--input",    required=False, default=None, help="Path to input image PNG (one-shot mode)")
    parser.add_argument("--mask",     required=False, default=None, help="Path to mask PNG (one-shot mode)")
    parser.add_argument("--output",   required=False, default=None, help="Path to write output PNG (one-shot mode)")
    parser.add_argument("--transformer",   required=True, help="Path to transformer component directory")
    parser.add_argument("--vae",           required=True, help="Path to VAE component directory")
    parser.add_argument("--text_encoder",  required=True, help="Path to text_encoder component directory")
    parser.add_argument("--text_encoder_2",required=True, help="Path to text_encoder_2 component directory")
    parser.add_argument("--tokenizer",     required=True, help="Path to tokenizer component directory")
    parser.add_argument("--tokenizer_2",   required=True, help="Path to tokenizer_2 component directory")
    parser.add_argument("--scheduler",     required=True, help="Path to scheduler component directory")
    parser.add_argument("--steps",    type=int,   default=20,  help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--seed",     type=int,   default=-1,  help="Seed (-1 for random)")
    parser.add_argument("--prompt", type=str, default="", help="Optional text prompt")
    parser.add_argument("--daemon", action="store_true", help="Run as persistent daemon, reading jobs from stdin")
    parser.add_argument("--log", type=str, default=None, help="Path to live log file (daemon mode)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_components(args):
    """Validate model component directories. Used by both one-shot and daemon modes."""
    errors = []
    component_args = {
        "transformer":    args.transformer,
        "vae":            args.vae,
        "text_encoder":   args.text_encoder,
        "text_encoder_2": args.text_encoder_2,
        "tokenizer":      args.tokenizer,
        "tokenizer_2":    args.tokenizer_2,
        "scheduler":      args.scheduler,
    }
    for name, path in component_args.items():
        if not os.path.isdir(path):
            errors.append("Component directory not found for {}: {}".format(name, path))

    if errors:
        for err in errors:
            print("ERROR: {}".format(err), file=sys.stderr)
        sys.exit(1)


def validate_job_paths(input_path, mask_path, output_path):
    """Validate input/mask files and output directory for a single job."""
    errors = []

    if not os.path.isfile(input_path):
        errors.append("Input file not found: {}".format(input_path))
    if not os.path.isfile(mask_path):
        errors.append("Mask file not found: {}".format(mask_path))

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            errors.append("Cannot create output directory: {}".format(e))

    if errors:
        for err in errors:
            _log("ERROR: {}".format(err))
        raise ValueError("Job path validation failed")

    _log("INFO: validate_job_paths passed.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(args, model_variant):
    print("INFO: Loading pipeline components.")
    print("INFO: This will take several minutes on first run.")
    _log("INFO: Loading pipeline components.")
    _log("INFO: Give the daemon a moment to load the model components...")

    try:
        import torch
        from diffusers import FluxFillPipeline, FluxTransformer2DModel, AutoencoderKL
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    except ImportError as e:
        print("ERROR: Failed to import required libraries: {}".format(e), file=sys.stderr)
        print("ERROR: Make sure the embedded Python environment was set up correctly by running setup.bat", file=sys.stderr)
        _log("ERROR: Failed to import required libraries: {}".format(e))
        _log("ERROR: Make sure the embedded Python environment was set up correctly by running setup.bat")
        sys.exit(2)

    try:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
            print("INFO: CUDA available. Using GPU: {}".format(
                torch.cuda.get_device_name(0)))
            print("INFO: VRAM available: {:.1f} GB".format(
                torch.cuda.get_device_properties(0).total_memory / 1e9))
        else:
            device = "cpu"
            dtype = torch.float32
            print("WARNING: CUDA not available. Running on CPU. This will be very slow.")
            _log("WARNING: CUDA not available. Running on CPU. This will be very slow.")

        _log("INFO: Using dtype: {}".format(dtype))

        _log("INFO: Loading transformer from: {}".format(args.transformer))
        _log("INFO: Model variant: {}".format(model_variant))
        if model_variant == "bf16":
            transformer = FluxTransformer2DModel.from_pretrained(
                args.transformer,
                torch_dtype=dtype,
                local_files_only=True,
            )
        elif model_variant == "fp8":
            try:
                from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            except ImportError as e:
                _log("ERROR: torchao not available for fp8 variant: {}".format(e))
                sys.exit(2)
            transformer = FluxTransformer2DModel.from_pretrained(
                args.transformer,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            _log("INFO: Applying fp8 quantization...")
            quantize_(transformer, float8_dynamic_activation_float8_weight())
            transformer.to("cuda")
        elif model_variant == "gguf":
            try:
                import glob
                from diffusers import GGUFQuantizationConfig
            except ImportError as e:
                _log("ERROR: Required library not available for gguf variant: {}".format(e))
                sys.exit(2)
            gguf_files = glob.glob(os.path.join(args.transformer, "*.gguf"))
            if not gguf_files:
                _log("ERROR: No .gguf file found in: {}".format(args.transformer))
                sys.exit(2)
            gguf_path = gguf_files[0]
            _log("INFO: Loading GGUF: {}".format(os.path.basename(gguf_path)))
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                config=args.transformer,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
        else:
            _log("ERROR: Unknown model_variant '{}'. Must be bf16, fp8, or gguf.".format(model_variant))
            sys.exit(2)

        _log("INFO: Loading VAE from: {}".format(args.vae))
        vae = AutoencoderKL.from_pretrained(
            args.vae,
            torch_dtype=dtype,
            local_files_only=True,
        )

        _log("INFO: Loading text_encoder from: {}".format(args.text_encoder))
        text_encoder = CLIPTextModel.from_pretrained(
            args.text_encoder,
            torch_dtype=dtype,
            local_files_only=True,
        )

        _log("INFO: Loading text_encoder_2 from: {}".format(args.text_encoder_2))
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.text_encoder_2,
            torch_dtype=dtype,
            local_files_only=True,
        )

        _log("INFO: Loading tokenizer from: {}".format(args.tokenizer))
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )

        _log("INFO: Loading tokenizer_2 from: {}".format(args.tokenizer_2))
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            args.tokenizer_2,
            local_files_only=True,
        )

        _log("INFO: Loading scheduler from: {}".format(args.scheduler))
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.scheduler,
            local_files_only=True,
        )

        _log("INFO: Assembling pipeline.")
        pipe = FluxFillPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        _log("INFO: Pipeline assembled. Moving to device: {}".format(device))
        _log("INFO: It may take several minutes to move the model to the device.")
        pipe = pipe.to(device)
        _log("INFO: Pipeline moved to device.")

        if device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 16.0:
                _log("INFO: VRAM under 16GB, enabling attention slicing.")
                pipe.enable_attention_slicing()

        _log("INFO: Pipeline loaded successfully.")
        return pipe

    except Exception as e:
        print("ERROR: Failed to load pipeline.", file=sys.stderr)
        print("ERROR: {}".format(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(input_path, mask_path):
    try:
        from PIL import Image
    except ImportError as e:
        print("ERROR: Pillow not available: {}".format(e), file=sys.stderr)
        sys.exit(3)

    _log("INFO: Loading input image: {}".format(input_path))
    input_img = Image.open(input_path).convert("RGB")

    _log("INFO: Loading mask image: {}".format(mask_path))
    mask_img = Image.open(mask_path).convert("L")

    if mask_img.size != input_img.size:
        _log("WARNING: Mask size {} does not match input {}. Resizing mask.".format(
            mask_img.size, input_img.size))
        mask_img = mask_img.resize(input_img.size, Image.LANCZOS)

    _log("INFO: Image size: {}x{}".format(input_img.width, input_img.height))
    return input_img, mask_img


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(pipe, input_img, mask_img, args):
    try:
        import torch

        if args.seed == -1:
            seed = random.randint(0, 2 ** 32 - 1)
            _log("INFO: Random seed: {}".format(seed))
        else:
            seed = args.seed
            _log("INFO: Using seed: {}".format(seed))

        seed_file = args.output + ".seed"
        with open(seed_file, "w") as f:
            f.write(str(seed))

        device = pipe.transformer.device
        generator = torch.Generator(device=device).manual_seed(seed)

        _log("INFO: Starting inference...")
        _log("INFO: Steps: {}  Guidance: {}".format(args.steps, args.guidance))

        total_steps = args.steps
        step_times = []

        def step_callback(pipeline, step_index, timestep, callback_kwargs):
            now = time.time()
            step_times.append(now)
            completed = step_index + 1
            pct = int((completed / total_steps) * 100)
            bar_filled = int((completed / total_steps) * 10)
            bar = "#" * bar_filled + " " * (10 - bar_filled)

            if len(step_times) >= 2:
                avg_step = (step_times[-1] - step_times[0]) / (len(step_times) - 1)
                elapsed = step_times[-1] - step_times[0]
                remaining = avg_step * (total_steps - completed)
                elapsed_str  = "{:02d}:{:05.2f}".format(int(elapsed // 60), elapsed % 60)
                remaining_str = "{:02d}:{:05.2f}".format(int(remaining // 60), remaining % 60)
                per_it = "{:.2f}s/it".format(avg_step)
                msg = "INFO: {}%|{}| {}/{} [{}<{}, {}]".format(
                    pct, bar, completed, total_steps,
                    elapsed_str, remaining_str, per_it)
            else:
                msg = "INFO: {}%|{}| {}/{}".format(pct, bar, completed, total_steps)
            
            if _log_path:
                _log(msg)

            return callback_kwargs

        result = pipe(
            prompt=args.prompt,
            image=input_img,
            mask_image=mask_img,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            height=input_img.height,
            width=input_img.width,
            callback_on_step_end=step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        _log("INFO: Inference complete.")
        return result, seed

    except Exception as e:
        _log("ERROR: Inference failed.")
        _log("ERROR: {}".format(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Output - 16-bit PNG writer
# ---------------------------------------------------------------------------

def save_16bit_png(rgb_array_uint16, output_path, metadata=None):
    """
    Write a uint16 numpy array (H, W, 3) as a 16-bit RGB PNG.
    Written manually to avoid Pillow's inconsistent 16-bit RGB support on Windows.
    Falls back to 8-bit if anything goes wrong.
    """
    import numpy as np
    import zlib
    import struct

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    try:
        h, w, c = rgb_array_uint16.shape
        assert c == 3
        assert rgb_array_uint16.dtype == np.uint16

        sig  = b"\x89PNG\r\n\x1a\n"
        ihdr = make_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))

        text_chunks = b""
        if metadata:
            for key, value in metadata.items():
                text_data = "{}\x00{}".format(key, value).encode("latin-1")
                text_chunks += make_chunk(b"tEXt", text_data)

        raw_rows = []
        for row in rgb_array_uint16:
            raw_rows.append(b"\x00" + row.astype(">u2").tobytes())

        idat = make_chunk(b"IDAT", zlib.compress(b"".join(raw_rows), 6))
        iend = make_chunk(b"IEND", b"")

        with open(output_path, "wb") as f:
            f.write(sig + ihdr + text_chunks + idat + iend)

        print("INFO: Saved as 16-bit PNG. text_chunks bytes: {}".format(len(text_chunks)))

    except Exception as e:
        print("WARNING: 16-bit PNG write failed ({}), falling back to 8-bit.".format(e))
        from PIL import Image
        arr_8 = (rgb_array_uint16 // 257).astype("uint8")
        Image.fromarray(arr_8, mode="RGB").save(output_path)
        print("INFO: Saved as 8-bit PNG (fallback).")


def save_output(result_img, output_path, seed, steps, guidance, prompt):
    try:
        import numpy as np
        _log("INFO: Saving output to: {}".format(output_path))
        arr = np.array(result_img.convert("RGB"))
        arr_16 = arr.astype(np.uint16) * 257
        metadata = {
            "seed":     str(seed),
            "steps":    str(steps),
            "guidance": str(guidance),
            "prompt":   prompt if prompt else "",
        }
        _log("INFO: Metadata written: {}".format(metadata))
        save_16bit_png(arr_16, output_path, metadata=metadata)
        _log("INFO: Output saved successfully.")
    except Exception as e:
        _log("ERROR: Failed to save output.")
        _log("ERROR: {}".format(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(4)


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------

def delete_temp_inputs(input_path, mask_path):
    for path in [input_path, mask_path]:
        try:
            if os.path.isfile(path):
                os.remove(path)
                _log("INFO: Deleted temp file: {}".format(path))
        except Exception as e:
            _log("WARNING: Could not delete temp file {}: {}".format(path, e))


# ---------------------------------------------------------------------------
# Daemon job loop
# ---------------------------------------------------------------------------

def run_daemon(pipe):
    """
    Persistent loop. Reads JSON job dicts from stdin one line at a time.
    Signals READY once after model load, then processes jobs until shutdown
    or stdin closes (Nuke exited).
    """
    import sys

    print("DBFLUXFILL_READY")
    sys.stdout.flush()

    _log("INFO: Daemon ready, waiting for jobs.")
    sys.stdout.flush()

    for raw in iter(sys.stdin.readline, ""):
        _log("INFO: Received line from stdin: {}".format(repr(raw)))
        raw = raw.strip()
        _log("INFO: Stripped line: {}".format(repr(raw)))
        if not raw:
            continue

        try:
            job = json.loads(raw)
            _log("INFO: json loaded successfully: {}".format(job))
        except Exception as e:
            _log("ERROR: Could not parse job JSON: {}".format(e))
            sys.stdout.flush()
            continue

        if job.get("shutdown"):
            _log("INFO: Shutdown message received. Exiting.")
            sys.stdout.flush()
            break

        # Validate required keys
        _log("INFO: About to validate job keys.")
        required = ["input", "mask", "output", "steps", "guidance", "seed", "prompt"]
        missing  = [k for k in required if k not in job]
        if missing:
            result = {"success": False, "error": "Missing keys: {}".format(missing)}
            print("DBFLUXFILL_RESULT:{}".format(json.dumps(result)))
            sys.stdout.flush()
            continue

        # Validate paths for this job
        try:
            validate_job_paths(job["input"], job["mask"], job["output"])
        except ValueError as e:
            result = {"success": False, "error": "Job path validation failed, see stderr"}
            print("DBFLUXFILL_RESULT:{}".format(json.dumps(result)))
            sys.stdout.flush()
            continue

        # Run inference
        _log("INFO: Starting inference block.")
        try:
            _log("INFO: Calling load_images.")
            input_img, mask_img = load_images(job["input"], job["mask"])

            class _JobArgs(object):
                pass

            job_args         = _JobArgs()
            job_args.seed    = job["seed"]
            job_args.steps   = job["steps"]
            job_args.guidance= job["guidance"]
            job_args.prompt  = job["prompt"]
            job_args.output  = job["output"]

            result_img, resolved_seed = run_inference(pipe, input_img, mask_img, job_args)
            save_output(result_img, job["output"], resolved_seed,
                        job["steps"], job["guidance"], job["prompt"])
            delete_temp_inputs(job["input"], job["mask"])
            seed_file = job["output"] + ".seed"
            if os.path.isfile(seed_file):
                try:
                    os.remove(seed_file)
                except Exception:
                    pass
            result = {"success": True, "seed": resolved_seed}


        except SystemExit as e:
            result = {"success": False, "error": "Runner exited with code {}".format(e.code)}
            _log("ERROR: {}".format(result["error"]))
        except Exception as e:
            result = {"success": False, "error": str(e)}
            _log("ERROR: {}".format(str(e)))
            traceback.print_exc()

        _log("INFO: Job complete. Check Nuke. Success={}".format(result.get("success")))
        _log("INFO: DO NOT CLOSE POWERSHELL WINDOW. The model is still loaded in VRAM.")
        _log('INFO: Use the "Unload Model" button to free resources and clean up log files when done.')
        _log('INFO: Clicking "Unload Model" will close this window as well.')
        print("DBFLUXFILL_RESULT:{}".format(json.dumps(result)))
        sys.stdout.flush()

    print("INFO: Daemon exiting.")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("INFO: flux_runner.py starting.")
    args = parse_args()

    # Read config once at startup
    config = load_config()
    model_variant = config.get("model_variant", "bf16")
    _log("INFO: model_variant from config.json: {}".format(model_variant))

    print("INFO: Arguments received:")
    print("  daemon:   {}".format(args.daemon))
    if not args.daemon:
        print("  input:    {}".format(args.input))
        print("  mask:     {}".format(args.mask))
        print("  output:   {}".format(args.output))
    print("  transformer:    {}".format(args.transformer))
    print("  vae:            {}".format(args.vae))
    print("  text_encoder:   {}".format(args.text_encoder))
    print("  text_encoder_2: {}".format(args.text_encoder_2))
    print("  tokenizer:      {}".format(args.tokenizer))
    print("  tokenizer_2:    {}".format(args.tokenizer_2))
    print("  scheduler:      {}".format(args.scheduler))
    if not args.daemon:
        print("  steps:    {}".format(args.steps))
        print("  guidance: {}".format(args.guidance))
        print("  seed:     {}".format(args.seed))
        print("  prompt:   {}".format(args.prompt))

    if args.daemon:
        os.environ["DIFFUSERS_DISABLE_TQDM"] = "1"
        _log_path = args.log
        validate_components(args)
        pipe = load_pipeline(args, model_variant)
        run_daemon(pipe)

    else:
        # One-shot mode: original behaviour
        validate_components(args)
        validate_job_paths(args.input, args.mask, args.output)
        pipe = load_pipeline(args, model_variant)
        input_img, mask_img = load_images(args.input, args.mask)
        result, resolved_seed = run_inference(pipe, input_img, mask_img, args)
        save_output(result, args.output, resolved_seed, args.steps, args.guidance, args.prompt)
        delete_temp_inputs(args.input, args.mask)
        print("INFO: flux_runner.py finished successfully.")

    sys.exit(0)