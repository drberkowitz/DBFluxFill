# DBFluxFill - FLUX.1 Fill Dev inside Nuke

DBFluxFill is a Nuke gizmo that brings [FLUX.1 Fill Dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) inpainting directly into your Nuke workflow. Connect an image and hit Generate. The result drops back into your node graph as a Read node, ready to comp.

It was built specifically for Nuke artists who may have never used ComfyUI, are not familar with that much Python, or don't want to learn another tool. The installer walks the user through everything, downloads the model components, sets up an embedded Python environment, has a Nuke Indie option, and even helps you set up your init.py file.

**License note:** DBFluxFill is a free, open-source tool. The underlying FLUX.1 Fill Dev model is subject to the [FLUX.1 [dev] License](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Model Variants](#model-variants)
- [Using DBFluxFill](#using-dbfluxfill)
- [Manual Model Setup](#manual-model-setup)
- [Important Notes](#important-notes)
- [Acknowledgments](#acknowledgments)

---

## Requirements

**Platform:** Windows 10 or 11 only

**Nuke:** Nuke 13.2v8 or later (built for 13.2, but tested on later versions)

**GPU:** Nvidia GPU with CUDA 12.4+ drivers required
- The installer checks this on launch and will warn you if your drivers are out of date
- [Download latest Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

**VRAM:** Depends on your chosen model variant (see [Model Variants](#model-variants) below)

**Disk space:** 23 to 35 GB free depending on model variant

**Hugging Face account:** Required. You must accept the FLUX.1 Fill Dev license on Hugging Face before downloading. [Accept the license here](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).

---

## Installation

### Step 1 — Download and place the folder

Click the green `<> Code` button at the top of this page. Select `Download ZIP` and unzip the downloaded folder.

Rename `DBFluxFill-main` to `DBFluxFill` and copy it into your `.nuke` directory:

```
C:\Users\YourName\.nuke\DBFluxFill\
```

If you are not sure where your `.nuke` folder is, open Nuke and run this in the Script Editor:

```python
print(nuke.pluginPath())
```

Your `.nuke` folder will be one of the paths listed.

### Step 2 — Run the installer

Inside the `DBFluxFill` folder, double-click `setup.bat`.

This launches the DBFluxFill installer, a step-by-step GUI that handles everything else:

- Downloading and installing Python 3.11 if necessary
- Setting up an embedded Python environment
- Installing all dependencies (including the correct PyTorch CUDA build)
- Downloading your chosen model variant from Hugging Face
- Writing the configuration file

![The DBFluxFill installer welcome screen](https://github.com/user-attachments/assets/f67fafce-c5cc-40d2-852e-38bd8cb19cf9)

The installer has six screens:

**Welcome** — Shows system requirements and checks your GPU and CUDA version. If your drivers are too old or no Nvidia GPU is detected, the installer will warn you before you proceed.

**File Paths** — Sets default directories for temporary PNG/seed/log files and output renders as well as default file name practice. Pre-populated with sensible defaults based on your current Nuke project. Each field supports Nuke TCL expressions. You can also set Nuke Indie mode here.

**Model Variant** — Choose which version of FLUX.1 Fill Dev to download. See [Model Variants](#model-variants) below for a full comparison.

**Model Setup** — Enter your Hugging Face token and choose where to store the models. If you prefer to download components manually, there is a link to the manual setup path from this screen.

**Installing** — The installer sets up the embedded Python environment, installs dependencies, and downloads the model. This will take a while depending on your internet speed. A live log shows progress.

**Done** — Shows a summary of what was installed and displays the code snippet you need to add to your `init.py`. See Step 3 below.

### Step 3 — Update your init.py

The installer's Done screen displays a code snippet. Add it to your `.nuke/init.py` file. It will look like this:

```python
import DBFluxFill
nuke.pluginAddPath('./DBFluxFill')
```

If you do not have an `init.py` yet, the Done screen offers to create one for you. The installer also has a button to open your `.nuke` folder in Explorer so you can find the file easily.

Once this is done, restart Nuke. DBFluxFill will appear in the toolbar.

---

## Model Variants

Three versions of FLUX.1 Fill Dev are supported. All three require you to accept the FLUX.1 Fill Dev license on Hugging Face before downloading.

| | bf16 | fp8 | gguf |
|---|---|---|---|
| **VRAM** | 16 GB+ recommended | 12 GB+ recommended | 10 GB+ recommended |
| **Disk** | ~34.3 GB | ~22.3 GB | ~23.1 GB |
| **Speed** | Baseline | Faster | Conditional |
| **Quality** | Full | Slight tradeoff | Slight tradeoff |
| **Source** | Black Forest Labs (official) | AlekseyCalvin + BFL components | YarvixPA Q5_1 + BFL components |

### bf16

The official FLUX.1 Fill Dev release in full diffusers format. Highest quality, highest VRAM requirement. Best choice if you have 16 GB+ VRAM and need the best quality.

- HuggingFace: [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)

### fp8

Uses the fp8-quantized AlekseyCalvin repo modified only by a small config.json change. Noticeably faster than bf16 with a minor quality tradeoff. Requires an Ampere GPU or newer (RTX 30xx+).

The installer applies a required `config.json` patch to the transformer after download automatically.

- HuggingFace: [AlekseyCalvin/FluxFillDev_fp8_Diffusers](https://huggingface.co/AlekseyCalvin/FluxFillDev_fp8_Diffusers)

### gguf

Uses a GGUF Q5_1 quantized transformer from YarvixPA with the official BFL VAE, encoders, and scheduler. Lowest VRAM requirement of the three variants. A good option for 10-12 GB and older cards.

The installer applies a required `config.json` patch and places the `.gguf` file correctly.

- HuggingFace: [YarvixPA/FLUX.1-Fill-dev-GGUF](https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF)

*To switch variants later, re-run `setup.bat` and select a different variant. The installer will update dependencies and rewrite `config.json`. If you want to free the disk space used by the previous variant, delete the models folder before re-running.*

---

## Manual Model Setup

If you prefer to download model components yourself rather than using a Hugging Face token, select **"I want to configure model components manually instead"** on the Model Setup screen of the installer.

The manual setup screen lists every component required, with direct links to the correct Hugging Face repositories and approximate file sizes. You point the installer at your local directories for each component and it writes the config accordingly.

### Components by variant

**bf16 components:**

All components come from [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev). The expected folder structure inside the models directory is:

```
models/
    transformer/
    vae/
    text_encoder/
    text_encoder_2/
    tokenizer/
    tokenizer_2/
    scheduler/
```

**fp8 components:**

All components from [AlekseyCalvin/FluxFillDev_fp8_Diffusers](https://huggingface.co/AlekseyCalvin/FluxFillDev_fp8_Diffusers). The folder structure should match the above bf16 models directory.

After downloading, you must overwrite `transformer/config.json` with the one from the BFL repo at `black-forest-labs/FLUX.1-Fill-dev/transformer/config.json`. The manual setup screen shows a config.json row for this compontent.

**gguf components:**

- Transformer file: download `flux1-fill-dev-Q5_1.gguf` from [YarvixPA/FLUX.1-Fill-dev-GGUF](https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF) and place it in a `transformer/` folder
- `transformer/config.json`: copy from [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/tree/main/transformer) The manual setup screen shows a config.json row for this component.
- All other components: from [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)

---

## Using DBFluxFill

![DBFluxFill in the node graph](https://github.com/user-attachments/assets/b53c746a-9f9c-4298-84f9-f3db64cd3b34)

### Setting up the node

1. Add `DBFluxFill` from the toolbar to your node graph (or search in the node graph).
2. Connect your image to the **img** input and your mask to the **mask** input. You can embed an alpha into the image input if you don't want to use the mask input, the gizmo handles both.
3. Your image and mask will be automatically cropped and scaled to a valid FLUX resolution (multiples of 16, minimum 272x272). If your image is larger than 2048px or smaller than 512px on either axis, the gizmo will warn you before generating.

### Generating

![DBFluxFill properties panel showing main tab knobs](https://github.com/user-attachments/assets/3486f058-0aa5-4b31-9e97-3de7c7c52753)

**Main tab knobs:**

- **Prompt** — Describe what you want to fill the masked area with. Not needed usually, but can help steer the generated image.
- **Seed** — Set to `-1` for a random seed each run, or enter a specific value to reproduce a result. The Randomize button sets it to `-1`.
- **Steps** — Number of diffusion steps. Default is 20. More steps generally means more detail but slower generation.
- **Guidance** — Controls how closely the output follows your prompt. Default is 2.5.
- **Prev Selector** — Switches the node's viewer output between your input and the pre-processed version sent to FLUX. Useful for checking what the model will actually see.
- **Mask Overlay** — Overlays a red mask in the viewer so you can see coverage.
- **Frame Hold** — Which frame from your input is used for generation. The gizmo will default to the frame the viewer is on when you drop the node down.
- **Mask Invert** — Inverts which area is treated as the fill region.
- **Crop to Mask** — When enabled (default), the gizmo crops the working area to the bounding box of your mask plus padding, which is faster. When disabled, the full image is sent to the model.
- **Crop Padding** — Pixels of padding added around the mask bounding box when Crop to Mask is on. Default is 25.

**Paths tab:**

Defaults are from the config.json which is setup in the Installer.

- **Temp Dir** — Where intermediate PNG files are written during generation. Supports TCL expressions.
- **Output Dir** — Where the final output PNG is saved. Supports TCL expressions.
- **Output Name** — The filename for the output. All outputs are saved as 16-bit PNG files.

When you hit **Generate**, Nuke will open a command prompt window showing the inference progress. Nuke itself will be unresponsive until generation completes. This is expected behaviour (see [Important Notes](#important-notes)).

**Generate (Keep Loaded)** opens a persistent PowerShell window and keeps the model in VRAM between generations. Subsequent runs skip the load step entirely. When you are done generating, click **Unload Model** to close the window, free VRAM, and clean up the temp log file.

When generation finishes, a Read node is dropped into your node graph with the result. It includes a **DBFluxFill Settings** tab showing the seed, steps, guidance, and prompt used, all read-only for reference.

![Result Read node dropped into node graph](https://github.com/user-attachments/assets/20c681d1-c27e-4e87-91a3-caaa832d7031)

![DBFluxFill settings tab showing seed, steps, guidence, and prompt used](https://github.com/user-attachments/assets/00d3a406-1603-4239-8e23-5af50cac4f72)

---

## Important Notes

**Nuke freezes during generation.** When you hit Generate, Nuke will be unresponsive until the model finishes. The command prompt window that opens shows inference progress. This is a subprocess architecture tradeoff. The model runs in an isolated Python environment to keep its memory separate from Nuke's. Use **Generate (Keep Loaded)** to keep the model in VRAM between generations, skipping the load time on subsequent runs.

**Windows only** The tool is designed and tested on Windows. macOS and Linux are not supported.

**config.json** in the folder also stores your temp directory, output directory, and output filename defaults set during installation. These can be edited directly in `config.json` at any time without re-running the installer. The file lives at `.nuke/DBFluxFill/config.json`

**Nuke 13.2 and later.** Designed for Nuke 13.2v8. Tested on later Nuke versions.

**VRAM floors are recommendations, not hard limits.** Generation may still run below the recommended VRAM amounts depending on your image resolution, but expect slower performance or out-of-memory errors at very high resolutions.

**Resolution limits.** The gizmo will warn you if your input is larger than 2048px or smaller than 512px on either axis. Very large inputs are not recommended and may exceed your VRAM. Very small inputs may produce poor unpredictable results.

**fp8 requires Ampere GPU or newer.** The fp8 variant uses float8 quantization which requires an RTX 30xx series card or newer. It will not run on Turing (RTX 20xx) or older.

**Nuke Indie.** DBFluxFill supports Nuke Indie. Indie mode is set during installation and affects how certain dialogs and file operations work inside Nuke. If you are on Indie, make sure to check the Indie mode box on the File Paths screen during install. This flag can be changed in the config.json as well.

---

## Acknowledgments

**FLUX.1 Fill Dev** — [Black Forest Labs](https://blackforestlabs.ai/). The model this tool is built around.

**fp8 repo** — [AlekseyCalvin](https://huggingface.co/AlekseyCalvin) for the fp8 FLUX Fill option.

**GGUF transformer** — [YarvixPA](https://huggingface.co/YarvixPA) for the FLUX.1 Fill Dev GGUF quantization.

**diffusers** — [Hugging Face](https://huggingface.co/docs/diffusers) for the inference pipeline that powers the backend.

---

***DBFluxFill** was developed by Daniel Berkowitz. The majority of the code was written with the assistance of [Claude](https://claude.ai)*
