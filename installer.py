"""
DBFluxFill Installer
Wizard-style setup for DBFluxFill Nuke gizmo.
Run via setup.bat

Screens:
    0 - Welcome
    1 - File Paths
    2 - Model Setup (HF token path)
    3 - Installing (progress + log)
    4 - Done
    5 - Manual Model Setup
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE  = "DBFluxFill Setup"

# Default path values shown in the file paths screen
DEFAULT_TEMP_DIR    = "[file dirname [value root.name]]"
DEFAULT_OUTPUT_DIR  = "[file dirname [value root.name]]"
DEFAULT_OUTPUT_NAME = "[file rootname [file tail [value root.name]]]-fluxfill-[clock format [clock seconds] -format %Y%m%d_%H%M%S]"

# Derive the gizmo root from installer location: .nuke/DBFluxFill/
GIZMO_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(GIZMO_DIR, "models")
PYTHON_DIR  = os.path.join(GIZMO_DIR, "python")
CONFIG_PATH = os.path.join(GIZMO_DIR, "config.json")

# TCL snippets shown in the path twirl-downs
# Each entry: (display_label, tcl_snippet)
TCL_TEMP_SNIPPETS = [
    ("Same folder as script",    "[file dirname [value root.name]]"),
    ("One level up, in /tmp",    "[file dirname [file dirname [value root.name]]]/tmp"),
    ("tmp subfolder",            "[file dirname [value root.name]]/tmp"),
]

TCL_OUTPUT_DIR_SNIPPETS = [
    ("Same folder as script",    "[file dirname [value root.name]]"),
    ("One level up, in /precomp",    "[file dirname [file dirname [value root.name]]]/precomp"),
    ("precomp subfolder",        "[file dirname [value root.name]]/precomp"),
]

TCL_OUTPUT_NAME_SNIPPETS = [
    ("Script name + timestamp",  "[file rootname [file tail [value root.name]]]-fluxfill-[clock format [clock seconds] -format %Y%m%d_%H%M%S]"),
    ("Script name only",         "[file rootname [file tail [value root.name]]]-fluxfill"),
    ("Generic + timestamp",      "fluxfill-[clock format [clock seconds] -format %Y%m%d_%H%M%S]"),
]

# Model components: (name, approx_size_label, subfolder)
MODEL_COMPONENTS = [
    ("transformer",    "~23.8 GB",  "transformer"),
    ("vae",            "~335 MB",   "vae"),
    ("text_encoder",   "~246 MB",   "text_encoder"),
    ("text_encoder_2", "~9.8 GB",   "text_encoder_2"),
    ("tokenizer",      "< 1 MB",    "tokenizer"),
    ("tokenizer_2",    "< 1 MB",    "tokenizer_2"),
    ("scheduler",      "< 1 MB",    "scheduler"),
]
MODEL_VARIANT_INFO = {
    "bf16": {
        "label":       "BF16",
        "release":     "OFFICIAL RELEASE",
        "size":        "~34.3 GB",
        "vram":        "16 GB+ recommended",
        "speed":       "Slower than FP8 - highest quality",
        "description": "The official full-precision release from Black Forest Labs. Highest quality output, largest download, highest VRAM requirement.",
        "source":      "Black Forest Labs (official)",
        "hf_url":      "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
        "hf_label":    "huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
    },
    "fp8": {
        "label":       "FP8",
        "release":     "COMMUNITY RELEASE",
        "size":        "~22.3 GB",
        "vram":        "12 GB+ recommended",
        "speed":       "Fastest - minor quality tradeoff",
        "description": "Community-converted FP8 safetensors file. Reduced VRAM and download size with minimal quality loss for most use cases. Not released by Black Forest Labs.",
        "source":      "Community (AlekseyCalvin)",
        "hf_url":      "https://huggingface.co/AlekseyCalvin/FluxFillDev_fp8_Diffusers",
        "hf_label":    "huggingface.co/AlekseyCalvin/FluxFillDev_fp8_Diffusers",
    },
    "gguf": {
        "label":       "GGUF",
        "release":     "COMMUNITY RELEASE",
        "size":        "~23.1 GB",
        "vram":        "10 GB+ recommended",
        "speed":       "Closer to BF16, possibly slower - moderate quality tradeoff",
        "description": "Community-quantized GGUF format (Q5_1). Smaller download and much lower VRAM usage than BF16 or FP8. More adaptable to graphics cards that are older and less capable. Not released by Black Forest Labs and quality is slightly reduced compared to official releases.",
        "source":      "Community (YarvixPA)",
        "hf_url":      "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF",
        "hf_label":    "huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF",
    },
}
TRANSFORMER_INFO = {
    "bf16": ("~23.8 GB", "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/tree/main/transformer"),
    "fp8":  ("~11.9 GB", "https://huggingface.co/AlekseyCalvin/FluxFillDev_fp8_Diffusers/tree/main/transformer"),
    "gguf": ("~12.7 GB", "https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q5_1.gguf"),
}


INIT_PY_SNIPPET = "import DBFluxFill\nnuke.pluginAddPath('./DBFluxFill')"

# ---------------------------------------------------------------------------
# Nvidia GPU detection
# ---------------------------------------------------------------------------

def _detect_cuda():
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_name = gpu_result.stdout.strip() if gpu_result.returncode == 0 else "Unknown Nvidia GPU"

            cuda_version = None
            for line in result.stdout.splitlines():
                if "CUDA Version:" in line:
                    parts = line.split("CUDA Version:")
                    cuda_version = parts[1].strip().split()[0]
                    break

            return gpu_name, cuda_version
    except Exception:
        pass
    return None, None

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)

        # -- Shared state vars (filled in by the user across screens) --
        self.temp_dir    = tk.StringVar(value=DEFAULT_TEMP_DIR)
        self.output_dir  = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_name = tk.StringVar(value=DEFAULT_OUTPUT_NAME)
        self.indie_mode  = tk.BooleanVar(value=False)
        self.model_dir   = tk.StringVar(value=MODELS_DIR.replace("\\", "/"))
        self.hf_token    = tk.StringVar()
        self.model_variant = tk.StringVar(value="bf16")
        self.manual_mode = False
        # Manual mode: one StringVar per component path
        self.manual_paths = {
            comp[0]: tk.StringVar(value=os.path.join(MODELS_DIR, comp[2]).replace("\\", "/"))
            for comp in MODEL_COMPONENTS
        }

        # -- Detect GPU info upfront for display in the welcome screen --
        self.gpu_name, self.cuda_version = _detect_cuda()

        # -- Header + step indicator --
        self._build_header()

        # -- Screen container --
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True, padx=24, pady=(8, 0))

        # -- Footer --
        self._build_footer()

        # -- Build all screens upfront, stack them --
        self.screens = {}
        self._build_screen_welcome()
        self._build_screen_paths()
        self._build_screen_variant()
        self._build_screen_model_setup()
        self._build_screen_installing()
        self._build_screen_done()
        self._build_screen_manual()

        # -- Navigation state --
        self.current_screen = None
        self.go_to(0)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    def _build_header(self):
        header = tk.Frame(self, pady=12)
        header.pack(fill="x", padx=24)

        tk.Label(header, text="DBFluxFill Setup", font=("Helvetica", 10)).pack(side="left")

        # Step dots - 5 dots for the main flow (screens 0-4)
        self.dot_frame = tk.Frame(header)
        self.dot_frame.pack(side="right")
        self.dots = []
        for i in range(6):
            c = tk.Canvas(self.dot_frame, width=10, height=10, highlightthickness=0)
            c.pack(side="left", padx=2)
            self.dots.append(c)

        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", padx=0)

    def _update_dots(self, active_index):
        """Fill dots: done=grey, active=black, future=light grey."""
        for i, c in enumerate(self.dots):
            c.delete("all")
            if i < active_index:
                color = "#888888"   # done
            elif i == active_index:
                color = "#111111"   # active
            else:
                color = "#cccccc"   # future
            c.create_oval(1, 1, 9, 9, fill=color, outline="")

    # -----------------------------------------------------------------------
    # Footer (Back / Continue buttons + step label)
    # -----------------------------------------------------------------------

    def _build_footer(self):
        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", padx=0)

        footer = tk.Frame(self, pady=10)
        footer.pack(fill="x", padx=24)

        self.step_label = tk.Label(footer, text="Step 1 of 5", fg="#888888", font=("Helvetica", 9))
        self.step_label.pack(side="left")

        self.btn_next = tk.Button(footer, text="Continue", width=12, command=self._on_next)
        self.btn_next.pack(side="right")

        self.btn_back = tk.Button(footer, text="Back", width=8, command=self._on_back)
        self.btn_back.pack(side="right", padx=(0, 6))


    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def go_to(self, index):
        self.update_idletasks()
        new_height = max(self.winfo_reqheight(), WINDOW_HEIGHT)
        self.geometry(f"{WINDOW_WIDTH}x{new_height}")
        
        # Hide current screen, show the new one, update header/footer.
        if self.current_screen is not None:
            self.screens[self.current_screen].pack_forget()

        self.current_screen = index
        self.screens[index].pack(fill="both", expand=True)
        if index == 5:
            self._refresh_done_screen()
        if index == 6:
            self._refresh_manual_screen()
        if index == 3:
            self._refresh_model_setup_screen()

        DOT_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 3}
        self._update_dots(DOT_MAP.get(index, index))

        labels = {
            0: "Step 1 of 6",
            1: "Step 2 of 6",
            2: "Step 3 of 6",
            3: "Step 4 of 6",
            4: "Step 5 of 6",
            5: "Step 6 of 6",
            6: "Step 4 of 6",
        }
        next_labels = {
            0: "Continue",
            1: "Continue",
            2: "Continue",
            3: "Start install",
            4: "",
            5: "Close",
            6: "Install",
        }

        self.step_label.config(text=labels.get(index, ""))
        self.btn_next.config(text=next_labels.get(index, "Continue"))
        self.btn_back.config(state="disabled" if index in (0, 4) else "normal")
        self.btn_next.config(state="disabled" if index == 4 else "normal")

    def _on_next(self):
        if self.current_screen == 0:
            self.go_to(1)
        elif self.current_screen == 1:
            self.go_to(2)
        elif self.current_screen == 2:
            self.go_to(3)
        elif self.current_screen == 3:
            self.manual_mode = False
            self.go_to(4)
            self._start_install()
        elif self.current_screen == 4:
            self.go_to(5)
        elif self.current_screen == 5:
            self.destroy()
        elif self.current_screen == 6:
            self.manual_mode = True
            self.go_to(4)
            self._start_install()

    def _on_back(self):
        if self.current_screen == 6:
            self.go_to(3)
        elif self.current_screen == 4:
            self.go_to(3)
        elif self.current_screen == 5:
            self.go_to(4)
        elif self.current_screen in (1, 2, 3):
            self.go_to(self.current_screen - 1)

    # -----------------------------------------------------------------------
    # Screen 0 - Welcome
    # -----------------------------------------------------------------------

    def _build_screen_welcome(self):
        f = tk.Frame(self.container)
        self.screens[0] = f

        tk.Label(f, text="Welcome to DBFluxFill", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text="AI inpainting with FLUX.1 Fill Dev inside Nuke!\n \nThis installer sets up an embedded Python environment, downloads or locates FLUX.1 Fill Dev model components, writes a config.json with your chosen paths and settings, and installs all required dependencies.",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 16))

        # Requirements grid
        req_frame = tk.Frame(f)
        req_frame.pack(fill="x", pady=(4, 24), padx=60)

        req_frame.columnconfigure(0, weight=1)
        req_frame.columnconfigure(1, weight=1)
        req_frame.columnconfigure(2, weight=1)
        req_frame.columnconfigure(3, weight=1)

        tk.Label(req_frame, text="REQUIREMENTS", font=("Helvetica", 8, "bold"), fg="#888888").grid(
           row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )

        cuda_ok = (
            self.gpu_name is not None
            and "nvidia" in self.gpu_name.lower()
            and self.cuda_version is not None
            and tuple(int(x) for x in self.cuda_version.split(".")[:2]) >= (12, 4)
        )

        cuda_display = f"{self.cuda_version}" if self.cuda_version else "Not detected"
        if self.cuda_version and tuple(int(x) for x in self.cuda_version.split(".")[:2]) < (12, 4):
            cuda_display = f"{self.cuda_version} (12.4+ required)"

        requirements = [
            ("Built for",                "Nuke 13.2v8"),
            ("Compatible with",          "Nuke 13.2 and later"),
            ("Python required",          "Python 3.11"),
            ("GPU VRAM (recommended)",   "12 GB+"),
            ("Disk space needed",        "~23 GB"),
            ("Platform",                 "Windows"),
            ("GPU",                      self.gpu_name if self.gpu_name else "Not detected"),
            ("CUDA version",             cuda_display if self.gpu_name else "N/A"),
        ]
        for row, (label, value) in enumerate(requirements):
            r = (row // 2) + 1
            c = (row % 2) * 2
            tk.Label(req_frame, 
                    text=label, 
                    fg="#888888", 
                    font=("Helvetica", 9)).grid(
                        row=r,
                        column=c,
                        sticky="w",
                        padx=(0, 4),
                        pady=2
                        )
            
            is_gpu_row  = label == "GPU"
            is_cuda_row = label == "CUDA version"
            color = "#111111"
            if is_gpu_row:
                color = "#2d6a2d" if (self.gpu_name and "nvidia" in self.gpu_name.lower()) else "#aa2222"
            if is_cuda_row:
                color = "#2d6a2d" if cuda_ok else "#aa2222"
            
            tk.Label(
                req_frame,
                text=value,
                font=("Helvetica", 9, "bold"),
                fg=color).grid(
                    row=r,
                    column=c+1,
                    sticky="w",
                    padx=(0, 24),
                    pady=2
                    )

        # Info note
        tk.Label(
            f,
            text="You'll be able to change file paths, the Nuke Indie flag, and model paths later by editing the config.json directly if needed.",
            justify="center",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
            padx=8,
            pady=24,
        ).pack(fill="x")

        if not cuda_ok:
            if self.gpu_name and "nvidia" in self.gpu_name.lower():
                warn_text = (
                    f"Your CUDA version ({self.cuda_version}) is below the required 12.4. Please update your Nvidia drivers to enable CUDA 12.4 support."
                )
                link_text = "Download latest Nvidia drivers: nvidia.com/download/index.aspx"
                link_url  = "https://www.nvidia.com/download/index.aspx"
            else:
                warn_text = (
                    "No Nvidia GPU detected. DBFluxFill requires an Nvidia GPU with CUDA 12.4 or later. AMD and Intel GPUs are not supported. Generation on CPU is possible but will be extremely slow (hours per image)."
                )
                link_text = "Download Nvidia drivers: nvidia.com/download/index.aspx"
                link_url  = "https://www.nvidia.com/download/index.aspx"

            warn_frame = tk.Frame(f, bg="#fff0f0", padx=10, pady=8)
            warn_frame.pack(fill="x", pady=(0, 8))
            tk.Label(
                warn_frame,
                text=warn_text,
                wraplength=WINDOW_WIDTH - 80,
                justify="left",
                bg="#fff0f0",
                fg="#aa2222",
                font=("Helvetica", 9, "bold"),
            ).pack(anchor="w")
            driver_link = tk.Label(
                warn_frame,
                text=link_text,
                bg="#fff0f0",
                fg="#0066cc",
                cursor="hand2",
                font=("Helvetica", 9, "underline"),
            )
            driver_link.pack(anchor="w", pady=(4, 0))
            driver_link.bind("<Button-1>", lambda e: self._open_url(link_url))
            self.after(0, lambda: self.btn_next.config(state="disabled"))


    # -----------------------------------------------------------------------
    # Screen 1 - File Paths
    # -----------------------------------------------------------------------

    def _build_screen_paths(self):
        f = tk.Frame(self.container)
        self.screens[1] = f

        tk.Label(f, text="File paths", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text="Where should DBFluxFill write temp files and outputs?\nTemp files (seed, image, log) are deleted after each generation. These settings are changeable later by editing the config.json.",
            justify="left",
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 12))

        # Helper: builds one path row + TCL twirl-down
        def path_row(parent, label_text, hint_text, string_var, snippets, browse=True):
            tk.Label(parent, text=label_text, font=("Helvetica", 10, "bold")).pack(anchor="w")
            tk.Label(parent, text=hint_text, fg="#888888", font=("Helvetica", 9)).pack(anchor="w")
            row = tk.Frame(parent)
            row.pack(fill="x", pady=(2, 0))
            entry = tk.Entry(row, textvariable=string_var, font=("Helvetica", 9))
            entry.pack(side="left", fill="x", expand=True)
            _build_tcl_twirl(parent, string_var, snippets)
            tk.Frame(parent, height=10).pack()   # spacer


        def _build_tcl_twirl(parent, string_var, snippets):
            """Collapsible frame listing TCL snippet options."""
            toggle_var = tk.BooleanVar(value=False)
            toggle_btn = tk.Label(parent, text="▶  TCL snippets", fg="#888888",
                                  cursor="hand2", font=("Helvetica", 9))
            toggle_btn.pack(anchor="w")
            snippet_frame = tk.Frame(parent, bg="#f5f5f5", padx=8, pady=4)
            # snippet_frame starts hidden

            def toggle():
                if toggle_var.get():
                    snippet_frame.pack_forget()
                    toggle_btn.config(text="▶  TCL snippets")
                    toggle_var.set(False)
                else:
                    snippet_frame.pack(fill="x", pady=(2, 0), after=toggle_btn)
                    toggle_btn.config(text="▼  TCL snippets")
                    toggle_var.set(True)
                self.update_idletasks()
                new_height = max(self.winfo_reqheight(), WINDOW_HEIGHT)
                self.geometry(f"{WINDOW_WIDTH}x{new_height}")

            toggle_btn.bind("<Button-1>", lambda e: toggle())

            for desc, snippet in snippets:
                row = tk.Frame(snippet_frame, bg="#f5f5f5")
                row.pack(fill="x", pady=1)

                t = tk.Text(
                    row,
                    font=("Courier", 8),
                    bg="#f5f5f5",
                    fg="#222222",
                    bd=0,
                    highlightthickness=0,
                    height=1,
                    wrap="none",
                )
                t.insert("1.0", snippet)
                t.config(state="disabled")
                t.pack(side="left", fill="x", expand=True)

                tk.Label(row, text=desc, font=("Helvetica", 8), bg="#f5f5f5",
                        fg="#888888").pack(side="left", padx=(6, 6))

        path_row(f, "Temp directory",
                 "Seed files, temp PNG images, and logs are written here.",
                 self.temp_dir, TCL_TEMP_SNIPPETS)

        path_row(f, "Output directory",
                 "Generated PNG files are saved here.",
                 self.output_dir, TCL_OUTPUT_DIR_SNIPPETS)

        path_row(f, "Output filename",
                 "No extension needed, all DBFluxFill outputs are 16-bit PNG files.",
                 self.output_name, TCL_OUTPUT_NAME_SNIPPETS)

        # Nuke Indie toggle
        sep = ttk.Separator(f, orient="horizontal")
        sep.pack(fill="x", pady=(4, 6))
        indie_row = tk.Frame(f)
        indie_row.pack(fill="x", pady=(0, 12))
        tk.Checkbutton(indie_row, text="Nuke Indie mode",
                       variable=self.indie_mode, font=("Helvetica", 10)).pack(side="left")
        tk.Label(indie_row, text="Enable if using Nuke Indie.",
                 fg="#888888", font=("Helvetica", 9)).pack(side="left", padx=(8, 0))


    # -----------------------------------------------------------------------
    # Screen 2 - Model Variant Selection
    # -----------------------------------------------------------------------

    def _build_screen_variant(self):
        f = tk.Frame(self.container)
        self.screens[2] = f

        tk.Label(f, text="Model variant", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text='Choose which variant of FLUX.1 Fill Dev to download and use. This affects download size, VRAM usage, and generation speed.\nYou can always run the installer again to switch variants later if needed. Just select a different variant and the installer will download the model files and change the config json. Delete the "models" folder before running the setup again if you want to free up disk space from the previous variant.',
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 16))

        body = tk.Frame(f)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=180)
        body.columnconfigure(1, weight=1)

        # -- Left column: radio buttons --
        left = tk.Frame(body)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 24))

        tk.Label(left, text="SELECT VARIANT", font=("Helvetica", 8, "bold"), fg="#888888").pack(anchor="w", pady=(0, 8))

        for key in ("bf16", "fp8", "gguf"):
            info = MODEL_VARIANT_INFO[key]
            rb_frame = tk.Frame(left)
            rb_frame.pack(anchor="w", pady=(0, 8))
            tk.Radiobutton(
                rb_frame,
                text=info["label"],
                variable=self.model_variant,
                value=key,
                font=("Helvetica", 11, "bold"),
                command=self._refresh_variant_panel,
            ).pack(anchor="w")
            release_color = "#2d6a2d" if info["release"] == "OFFICIAL RELEASE" else "#888888"
            tk.Label(
                rb_frame,
                text=info["release"],
                font=("Helvetica", 7, "bold"),
                fg=release_color,
            ).pack(anchor="w", padx=(22, 0))

        # -- Right column: info panel --
        right = tk.Frame(body, bg="#f5f5f5", padx=16, pady=14)
        right.grid(row=0, column=1, sticky="nsew")
        self._variant_panel = right
        self._variant_labels = {}

        def info_row(key, label_text):
            row = tk.Frame(right, bg="#f5f5f5")
            row.pack(fill="x", pady=(0, 6))
            tk.Label(row, text=label_text, font=("Helvetica", 8, "bold"), fg="#888888",
                    bg="#f5f5f5", width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="", font=("Helvetica", 9), fg="#111111",
                        bg="#f5f5f5", anchor="w", justify="left", wraplength=360)
            lbl.pack(side="left", fill="x", expand=True)
            self._variant_labels[key] = lbl

        tk.Label(right, text="VARIANT DETAILS", font=("Helvetica", 8, "bold"),
                fg="#888888", bg="#f5f5f5").pack(anchor="w", pady=(0, 10))

        info_row("size",        "Size")
        info_row("vram",        "VRAM")
        info_row("speed",       "Speed")
        info_row("description", "About")
        info_row("source",      "Source")

        # HF link row
        link_row = tk.Frame(right, bg="#f5f5f5")
        link_row.pack(fill="x", pady=(0, 6))
        tk.Label(link_row, text="More info", font=("Helvetica", 8, "bold"), fg="#888888",
                bg="#f5f5f5", width=10, anchor="w").pack(side="left")
        self._variant_hf_link = tk.Label(link_row, text="", font=("Helvetica", 9, "underline"),
                                        fg="#0066cc", bg="#f5f5f5", cursor="hand2", anchor="w")
        self._variant_hf_link.pack(side="left")
        self._variant_hf_link.bind("<Button-1>", lambda e: self._open_url(
            MODEL_VARIANT_INFO[self.model_variant.get()]["hf_url"]
        ))

        # fp8 warning label (only shown for fp8 variant)
        self.fp8_warning_lbl = tk.Label(right, text="", font=("Helvetica", 9), fg="red",
                               bg="#f5f5f5", anchor="w", justify="left", wraplength=360)
        self.fp8_warning_lbl.pack(side="bottom", anchor="w", pady=(8, 0))

        self._refresh_variant_panel()

    def _refresh_variant_panel(self):
        info = MODEL_VARIANT_INFO[self.model_variant.get()]
        for key, lbl in self._variant_labels.items():
            lbl.config(text=info[key])
        self._variant_hf_link.config(text=info["hf_label"])
        msg = "FP8 (float8_e4m3fn) runs best on Nvidia 30-series (Ampere) or newer. VRAM and performance savings may vary with older graphics cards compared to BF16."
        self.fp8_warning_lbl.config(text=msg if self.model_variant.get() == "fp8" else "")


    # -----------------------------------------------------------------------
    # Screen 3 - Model Setup (HF token)
    # -----------------------------------------------------------------------

    def _build_screen_model_setup(self):
        f = tk.Frame(self.container)
        self.screens[3] = f

        tk.Label(f, text="Model setup", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text="Due to Black Forest Labs license restrictions, model components cannot be bundled with this tool. The installer can download them with a Hugging Face token, or you can choose to download them manually.",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 8))

        # Size note
        self._size_note_label = tk.Label(
            f,
            text="",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
        )
        self._size_note_label.pack(anchor="w", pady=(0, 10))

        # Model folder
        tk.Label(f, text="Model folder", font=("Helvetica", 10, "bold")).pack(anchor="w")
        mrow = tk.Frame(f)
        mrow.pack(fill="x", pady=(2, 0))
        tk.Entry(mrow, textvariable=self.model_dir, font=("Helvetica", 9)).pack(side="left", fill="x", expand=True)
        tk.Button(mrow, text="Browse", font=("Helvetica", 9),
                  command=self._browse_model_dir).pack(side="left", padx=(4, 0))
        tk.Label(
            f,
            text="Where do you want the model components to be downloaded?",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#888888",
            font=("Helvetica", 8),
        ).pack(anchor="w", pady=(0, 10))

        # HF token steps
        tk.Label(f, text="HOW TO GET A HUGGING FACE TOKEN", font=("Helvetica", 8), fg="#888888").pack(anchor="w", pady=(0, 4))
        self._steps_frame = tk.Frame(f)
        self._steps_frame.pack(fill="x")

        # Token entry
        tk.Label(f, text="Hugging Face token", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Entry(f, textvariable=self.hf_token, show="*", font=("Helvetica", 9)).pack(fill="x")
        tk.Label(
            f,
            text="Your token is not stored. It is used only to authenticate the download, then deleted from memory. It is never written to disk or config.json.",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#888888",
            font=("Helvetica", 9),
        ).pack(anchor="w", pady=(4, 8))

        # Manual link
        manual_lbl = tk.Label(f, text="I want to configure model components manually instead",
                              fg="#888888", cursor="hand2", font=("Helvetica", 9, "underline"))
        manual_lbl.pack(anchor="w")
        manual_lbl.bind("<Button-1>", lambda e: self.go_to(6))

    def _browse_model_dir(self):
        chosen = filedialog.askdirectory()
        if chosen:
            self.model_dir.set(chosen.replace("\\", "/"))

    def _refresh_model_setup_screen(self):
        info = MODEL_VARIANT_INFO[self.model_variant.get()]
        raw_size = info['size'].split()[0].replace('~', '')
        base_size = int(float(raw_size))
        total_required = (base_size + 9) // 5 * 5
        self._size_note_label.config(
            text=f"Download size: {info['size']} for all components. ~5 GB additional for the embedded Python environment.\n"
                 f"Ensure at least {total_required} GB of free disk space before continuing."
        )

        for widget in self._steps_frame.winfo_children():
            widget.destroy()

        steps = [
            ("Go to ", "huggingface.co/black-forest-labs/FLUX.1-Fill-dev", "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev", " sign in or create an account and accept the model license."),
            ("On ", "huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens", ' click "+Create new token" and set the token type to "Read".'),
            ('Name it something like "dbfluxfill-setup" so you can identify and revoke it later. Click "Create Token"', None, None, None),
            ('Copy the token (starts with "hf_...") and paste it below.', None, None, None),
        ]
        
        for i, step in enumerate(steps, 1):
            row = tk.Frame(self._steps_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{i}.", width=2, font=("Helvetica", 9, "bold")).pack(side="left")
            if isinstance(step, tuple) and step[1]:
                before, link_text, url, after = step
                inner = tk.Frame(row)
                inner.pack(side="left", anchor="w")
                tk.Label(inner, text=before, font=("Helvetica", 9), fg="#333333").pack(side="left")
                link = tk.Label(inner, text=link_text, font=("Helvetica", 9), fg="#0066cc", cursor="hand2")
                link.pack(side="left")
                link.bind("<Button-1>", lambda e, u=url: self._open_url(u))
                if after:
                    tk.Label(inner, text=after, font=("Helvetica", 9), fg="#333333").pack(side="left")
            else:
                text = step[0] if isinstance(step, tuple) else step
                tk.Label(row, text=text, font=("Helvetica", 9), fg="#333333",
                        justify="left", wraplength=WINDOW_WIDTH - 80).pack(side="left", anchor="w")


    # -----------------------------------------------------------------------
    # Screen 4 - Installing
    # -----------------------------------------------------------------------

    def _build_screen_installing(self):
        f = tk.Frame(self.container)
        self.screens[4] = f

        tk.Label(f, text="Installing", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text="Setting up the embedded Python environment, installing dependencies, and downloading model components. This may take a while.",
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(f, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(0, 8))

        self.log_text = tk.Text(f, height=14, font=("Courier", 9), state="disabled",
                                bg="#f5f5f5", relief="flat", borderwidth=1)
        self.log_text.pack(fill="both", expand=True)

        # Tag configs for coloured log lines - set up once here
        self.log_text.tag_config("ok",   foreground="#2d6a2d")
        self.log_text.tag_config("info", foreground="#111111")
        self.log_text.tag_config("err",  foreground="#aa2222")

        self.status_label = tk.Label(f, text="", fg="#888888", font=("Helvetica", 9))
        self.status_label.pack(anchor="w", pady=(4, 0))

    def _log(self, message, tag="info"):
        """Append a line to the log widget. tag: 'ok', 'info', or 'err'."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _start_install(self):
        """Kick off the install pipeline in a background thread."""
        import threading
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.progress_var.set(0)
        t = threading.Thread(target=self._install_worker, daemon=True)
        t.start()

    def _install_worker(self):
        """
        Runs in a background thread. Calls self._log() and self._set_progress()
        to update the UI. Never touches widgets directly.
        Steps:
            1. Set up embedded python enviroment
            2. Install pip dependencies
            3. Download models (token) or validate paths (manual)
            4. Write config.json
            5. Advance to done screen
        """
        try:
            self._step_ensure_python311()
            self._step_bootstrap_pip()
            self._step_install_deps()
            if self.manual_mode:
                self._step_validate_manual_paths()
            else:
                self._step_download_models()
            self._step_write_config()
            self._log("Setup complete. Press Continue to finish.", "ok")
            self.after(0, lambda: self.btn_next.config(state="normal", text="Continue"))
        except Exception as e:
            self._log(f"ERROR: {e}", "err")
            self._log("Setup did not complete. Check the log above.", "err")
            # Re-enable the back button so the user isn't stuck
            self.after(0, lambda: self.btn_back.config(state="normal"))

    def _set_progress(self, value):
        """Thread-safe progress bar update."""
        self.after(0, lambda: self.progress_var.set(value))

    def _set_status(self, text):
        """Thread-safe status label update."""
        self.after(0, lambda: self.status_label.config(text=text))
    
    def _step_ensure_python311(self):
        self._set_status("Locating embedded Python...")
        self._log("Checking for embedded Python 3.11...")

        python_exe = os.path.join(GIZMO_DIR, "python", "python.exe")
        if not os.path.isfile(python_exe):
            raise RuntimeError(
                "Embedded Python not found at:\n{}\n\n"
                "Please re-run setup.bat to download and unpack it.".format(python_exe)
            )

        self.python311_exe = [python_exe]
        self._log("Embedded Python found: {}".format(python_exe), "ok")
        self._set_progress(5)

    def _step_bootstrap_pip(self):
        self._set_status("Bootstrapping pip into embedded Python...")
        import subprocess, urllib.request, tempfile


        python_exe = self.python311_exe[0]
        pip_check = subprocess.run(
            [python_exe, "-m", "pip", "--version"],
            capture_output=True
        )
        if pip_check.returncode == 0:
            self._log("pip already available. Skipping get-pip.", "ok")
            self._set_progress(10)
            return

        self._log("pip not found. Downloading get-pip.py...")
        get_pip_url  = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_path = os.path.join(tempfile.gettempdir(), "get-pip.py")

        try:
            urllib.request.urlretrieve(get_pip_url, get_pip_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to download get-pip.py: {}\n\n"
                "To install manually:\n"
                "  1. Download https://bootstrap.pypa.io/get-pip.py\n"
                "  2. Place it anywhere on this machine\n"
                "  3. Run: {}  path\\to\\get-pip.py\n"
                "  4. Re-run setup.bat".format(e, python_exe)
            )

        self._log("Running get-pip.py...")
        result = subprocess.run(
            [python_exe, get_pip_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "get-pip.py failed:\n{}".format(result.stderr)
            )

        self._log("pip installed successfully.", "ok")
        self._set_progress(10)

    def _step_install_deps(self):
        self._set_status("Installing dependencies (this may take several minutes)...")
        self._log("Installing pip dependencies...")
        import subprocess

        pip_env = os.environ.copy()
        pip_env["PYTHONNOUSERSITE"] = "1"

        base_packages = [
            ["diffusers==0.37.1"],
            ["transformers==5.3.0"],
            ["accelerate==1.13.0"],
            ["safetensors==0.7.0"],
            ["Pillow==12.1.1"],
            ["numpy==2.4.3"],
            ["sentencepiece"],
            ["protobuf"],
        ]

        variant_packages = {
            "fp8":  [["torchao==0.9.0"]],
            "gguf": [["gguf>=0.10.0"]],
            "bf16": [],
        }

        packages = base_packages + variant_packages.get(self.model_variant.get(), [])

        python_exe = self.python311_exe[0]

        self._log("Upgrading pip...")
        proc = subprocess.Popen(
            [python_exe, "-m", "pip", "install", "--upgrade", "--no-user", "--no-warn-script-location", "pip"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=pip_env
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._log(f"    {line}")
        proc.wait()

        self._log("Installing Torch with CUDA support...")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "torch==2.6.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124", "--no-user", "--no-warn-script-location"],
            capture_output=True,
            env=pip_env
        )

        self._log(f"returncode: {result.returncode}", "info")
        self._log(f"stdout: {result.stdout.decode('utf-8', errors='replace')[:500]}", "info")
        self._log(f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}", "err")

        for i, pkg in enumerate(packages):
                self._log(f"  pip install {pkg[0]}")

                cmd = [python_exe, "-m", "pip", "install", "--no-user", "--no-warn-script-location"] + pkg

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=pip_env
                )

                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(f"    {line}")

                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to install {pkg[0]}. Check log above.")

                self._log(f"  installed {pkg[0]}", "ok")
                self._set_progress(10 + int((i + 1) / len(packages) * 40))

    def _step_download_models(self):
        self._set_status("Downloading model components from Hugging Face...")
        self._log("Starting model download...")
        import subprocess

        token = self.hf_token.get().strip()
        if not token:
            raise RuntimeError("No Hugging Face token provided.")

        python_exe     = self.python311_exe[0]
        download_script = os.path.join(GIZMO_DIR, "download_models.py")
        dest            = self.model_dir.get().strip()

        proc = subprocess.Popen(
            [python_exe, download_script, "--token", token, "--output", dest, "--variant", self.model_variant.get()],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._log(f"  {line}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Model download failed. Check log above.")

        self.after(0, lambda: self.hf_token.set(""))
        self._log("HF token cleared from memory.", "ok")
        self._set_progress(85)

    def _step_validate_manual_paths(self):
        self._set_status("Validating component paths...")
        self._log("Validating manual component paths...")
        missing = []
        for name, _, _ in MODEL_COMPONENTS:
            path = self.manual_paths[name].get().strip()
            if not os.path.isdir(path):
                missing.append(f"  {name}: {path}")
        if missing:
            raise RuntimeError("The following component folders were not found:\n" + "\n".join(missing))
        self._log("All component paths valid.", "ok")
        self._set_progress(85)

    def _step_write_config(self):
        self._set_status("Writing config.json...")
        self._log("Writing config.json...")
        import json

        if self.manual_mode:
            component_paths = {name: self.manual_paths[name].get().strip() for name, _, _ in MODEL_COMPONENTS}
        else:
            base = self.model_dir.get().strip()
            component_paths = {subfolder: f"{base}/{subfolder}" for _, _, subfolder in MODEL_COMPONENTS}

        config = {
            "nuke_indie": self.indie_mode.get(),
            "temp_dir":   self.temp_dir.get(),
            "output_dir": self.output_dir.get(),
            "output_name": self.output_name.get(),
            "model_variant":  self.model_variant.get(),
            "components": component_paths,
        }

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)

        self._log(f"config.json written to {CONFIG_PATH}", "ok")
        self._set_progress(100)

    # -----------------------------------------------------------------------
    # Screen 5 - Done
    # -----------------------------------------------------------------------

    def _build_screen_done(self):
        f = tk.Frame(self.container)
        self.screens[5] = f

        tk.Label(f, text="Setup complete", font=("Helvetica", 16)).pack(anchor="w", pady=(16, 4))
        tk.Label(
            f,
            text="Everything is configured. Here is a summary of what was written.",
            justify="left",
            fg="#555555",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(0, 12))

        summary_items = [
            ("Embedded Python ready",    "_done_lbl_python"),
            ("Model components ready", "_done_lbl_models"),
            ("config.json written",    "_done_lbl_config"),
        ]
        for title, attr in summary_items:
            row = tk.Frame(f)
            row.pack(fill="x", pady=2)
            tk.Label(row, text="✓", fg="#2d6a2d", font=("Helvetica", 11, "bold")).pack(side="left", padx=(0, 6))
            tk.Label(row, text=title, font=("Helvetica", 10, "bold")).pack(side="left")
            lbl = tk.Label(row, text="", fg="#888888", font=("Helvetica", 9))
            lbl.pack(side="left")
            setattr(self, attr, lbl)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=12)

        tk.Label(f, text="ADD TO YOUR INIT.PY", font=("Helvetica", 8), fg="#888888").pack(anchor="w")
        tk.Label(
            f,
            text='Add these two lines to your Nuke "init.py" file. If you don\'t have one yet, use the button below to create one.\nTo edit your "init.py", right click and open it in Notepad or your preferred code editor. Then paste the snippet and save.',
            justify="left",
            wraplength=WINDOW_WIDTH - 60,
            fg="#555555",
            font=("Helvetica", 9),
        ).pack(anchor="w", pady=(4, 6))

        # Code block
        code_frame = tk.Frame(f, bg="#f0f0f0", padx=10, pady=8)
        code_frame.pack(fill="x", pady=(0, 6))
        t = tk.Text(code_frame, font=("Courier", 10), bg="#f0f0f0", fg="#111111",
            relief="flat", bd=0, height=INIT_PY_SNIPPET.count("\n") + 1)
        t.insert("1.0", INIT_PY_SNIPPET)
        t.config(state="disabled")
        t.pack(side="left")
        tk.Button(code_frame, text="Copy", font=("Helvetica", 9),
                  command=self._copy_init_snippet).pack(side="right")

        # init.py buttons
        btn_row = tk.Frame(f)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="Open my init.py directory", font=("Helvetica", 9),
                  command=self._open_init_py).pack(side="left", padx=(0, 6))

    def _copy_init_snippet(self):
        """Copy the init.py snippet to clipboard."""
        self.clipboard_clear()
        self.clipboard_append(INIT_PY_SNIPPET)

    def _open_init_py(self):
        import subprocess
        nuke_dir   = os.path.dirname(GIZMO_DIR)
        init_path  = os.path.join(nuke_dir, "init.py")

        if not os.path.isfile(init_path):
            create = messagebox.askyesno(
                "No init.py found",
                f"No init.py was found at:\n{init_path}\n\nCreate one now?"
            )
            if create:
                os.makedirs(nuke_dir, exist_ok=True)
                with open(init_path, "w") as f:
                    f.write("")
            else:
                subprocess.Popen(r'explorer ' + nuke_dir)
                return
        
        subprocess.Popen(r'explorer /select,' + init_path)

    def _refresh_done_screen(self):
        """Update done screen summary labels with current install state."""
        models_detail = (
            "Model paths configured" if self.manual_mode
            else self.model_dir.get().strip()
        )
        self._done_lbl_python.config(text=f"  -  {PYTHON_DIR.replace(chr(92), '/')}")
        self._done_lbl_models.config(text=f"  -  {models_detail}")
        self._done_lbl_config.config(text=f"  -  {CONFIG_PATH.replace(chr(92), '/')}")

    # -----------------------------------------------------------------------
    # Screen 6 - Manual Model Setup
    # -----------------------------------------------------------------------

    def _build_screen_manual(self):
        f = tk.Frame(self.container)
        self.screens[6] = f

        segments = [
            ("Download all the files for each component from Hugging Face and place them in the folder shown, or point to an existing location. Be sure to accept the ", None),
            ("Black Forest Labs license", "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"),
            (" on Hugging Face before downloading.", None),
        ]

        desc_frame = tk.Frame(f)
        desc_frame.pack(anchor="w", pady=(0, 10), fill="x")

        t = tk.Text(
            desc_frame,
            font=("Helvetica", 10),
            fg="#555555",
            bg=desc_frame.cget("bg"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            height=2,
            cursor="arrow",
        )
        t.pack(fill="x", expand=True)

        for i, (text, link_url) in enumerate(segments):
            if link_url:
                tag = f"link_{i}"
                t.tag_config(tag, foreground="#0066cc", underline=True)
                t.tag_bind(tag, "<Button-1>", lambda e, u=link_url: self._open_url(u))
                t.tag_bind(tag, "<Enter>", lambda e: t.config(cursor="hand2"))
                t.tag_bind(tag, "<Leave>", lambda e: t.config(cursor="arrow"))
                t.insert("end", text, tag)
            else:
                t.insert("end", text)

        t.config(state="disabled")

        tk.Label(f,
                 text="These components may or may not match ComfyUI or other community tools, so it's recommended to download from Hugging Face if possible to ensure you have the correct files.",
                 font=("Helvetica", 10), fg="#555555", justify="left", wraplength=WINDOW_WIDTH - 60).pack(anchor="w", pady=(0, 10))

        # Column headers
        header = tk.Frame(f)
        header.pack(fill="x")
        tk.Label(header, text="Component",  font=("Helvetica", 9), fg="#888888", width=14, anchor="w").pack(side="left")
        tk.Label(header, text="Size",       font=("Helvetica", 9), fg="#888888", width=8,  anchor="w").pack(side="left")
        tk.Label(header, text="Folder path",font=("Helvetica", 9), fg="#888888",            anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(header, text="Link",       font=("Helvetica", 9), fg="#888888", width=6,  anchor="w").pack(side="left")

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=(2, 4))

        # One row per component
        self._manual_component_links = {}

        self._manual_config_json_row = None

        for name, size, subfolder in MODEL_COMPONENTS:
            row = tk.Frame(f)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name, font=("Helvetica", 9, "bold"), width=14, anchor="w").pack(side="left")
            size_lbl = tk.Label(row, text=size, font=("Helvetica", 9), width=8, anchor="w", fg="#888888")
            size_lbl.pack(side="left")
            tk.Entry(row, textvariable=self.manual_paths[name], font=("Helvetica", 8)).pack(side="left", fill="x", expand=True)
            tk.Button(row, text="Browse", font=("Helvetica", 9), command=lambda name=name: self._browse_component_dir(name)).pack(side="left", padx=(4, 0))
            link = tk.Label(row, text="HF →", fg="#2255aa", cursor="hand2", font=("Helvetica", 9, "underline"), width=6)
            link.pack(side="left", padx=(4, 0))
            self._manual_component_links[name] = (link, subfolder)
            if name == "vae":
                self._vae_row = row
            if name == "transformer":
                self._manual_transformer_size = size_lbl
                
                # config.json row (fp8 and gguf only)
                self._manual_config_json_row = tk.Frame(f)
                header_row = tk.Frame(self._manual_config_json_row)
                header_row.pack(fill="x")
                tk.Label(header_row, text="config.json", font=("Helvetica", 9, "bold"), width=14, anchor="w").pack(side="left")
                tk.Label(header_row, text="< 1 MB", font=("Helvetica", 9), width=8, anchor="w", fg="#888888").pack(side="left")
                
                self.config_json_lbl = tk.Label(
                    header_row, 
                    text="Place in the transformers folder above and overwrite existing config.json if necessary.",
                    font=("Helvetica", 8, "italic"),
                    fg="#666666",
                    anchor="w"
                )
                self.config_json_lbl.pack(side="left", padx=(4, 0))
                link = tk.Label(header_row, text="HF →", fg="#2255aa", cursor="hand2", font=("Helvetica", 9, "underline"), width=6)
                link.pack(side="right", padx=(4, 0))
                link.bind("<Button-1>", lambda e: self._open_url("https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/transformer/config.json"))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=(8, 4))

        back_lbl = tk.Label(f, text="Use a Hugging Face token instead",
                            fg="#888888", cursor="hand2", font=("Helvetica", 9, "underline"))
        back_lbl.pack(anchor="w")
        back_lbl.bind("<Button-1>", lambda e: self.go_to(3))

    def _open_url(self, url):
        """Open a URL in the default browser."""
        import webbrowser
        webbrowser.open(url)

    def _browse_component_dir(self, name):
        chosen = filedialog.askdirectory()
        if chosen:
            self.manual_paths[name].set(chosen.replace("\\", "/"))

    def _refresh_manual_screen(self):
        variant = self.model_variant.get()
        size, transformer_url = TRANSFORMER_INFO[variant]
        self._manual_transformer_size.config(text=size)

        base_url = MODEL_VARIANT_INFO[variant if variant == "fp8" else "bf16"]["hf_url"] + "/tree/main/"

        for name, (link, subfolder) in self._manual_component_links.items():
            url = transformer_url if name == "transformer" else base_url + subfolder
            link.bind("<Button-1>", lambda e, u=url: self._open_url(u))
        
        if self._manual_config_json_row is not None:
            if variant in ("fp8", "gguf"):
                self._manual_config_json_row.pack(fill="x", pady=2, before=self._vae_row)
            else:
                self._manual_config_json_row.pack_forget()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _preflight_checks():
    import platform

    # Windows version check
    if platform.system() != "Windows":
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unsupported platform", "DBFluxFill is only supported on Windows.")
        root.destroy()
        sys.exit(1)

    win_ver = tuple(int(x) for x in platform.version().split(".")[:2])
    if win_ver < (10, 0):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unsupported Windows version", "DBFluxFill requires Windows 10 or later.")
        root.destroy()
        sys.exit(1)

if __name__ == "__main__":
    _preflight_checks()
    app = InstallerApp()
    app.mainloop()