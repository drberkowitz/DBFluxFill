"""
callbacks.py
Nuke-side callbacks for DBFluxFill.
Handles the Generate button, knobChanged previews, and post-generate node dropping.

Python 3.7+ compatible.
"""

import os
import sys
import json
import subprocess
import threading
import traceback

import nuke


# ---------------------------------------------------------------------------
# Module-level daemon state
# ---------------------------------------------------------------------------


_daemon_process  = None              # Popen handle for the persistent runner
_daemon_lock     = threading.Lock()  # guards _daemon_process access
_daemon_ready    = threading.Event() # set when daemon prints READY to stdout
_job_done        = threading.Event() # set when daemon prints RESULT to stdout
_job_result      = {}                # populated by reader thread with last result
_reader_thread   = None              # stdout reader thread for current daemon
_daemon_log_path = None              # path to the live log file for this session
_log_window_proc = None              # Popen handle for the tail CMD window


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _get_gizmo_dir():
    """Find the DBFluxFill folder by searching nuke plugin paths."""
    for p in nuke.pluginPath():
        candidate = os.path.join(p, "DBFluxFill")
        if os.path.isdir(candidate):
            return candidate
    return os.path.dirname(os.path.abspath(__file__))

def _get_python_exe():
    """Return path to the embedded Python executable."""
    gizmo_dir = _get_gizmo_dir()
    return os.path.join(gizmo_dir, "python", "python.exe")


def _load_config():
    """Load config.json from the gizmo folder. Returns dict or empty dict on failure."""
    gizmo_dir   = _get_gizmo_dir()
    config_path = os.path.join(gizmo_dir, "config.json")
    if not os.path.isfile(config_path):
        nuke.message(
            "DBFluxFill: config.json not found.\n\n"
            "Please run setup.bat to configure the gizmo.\n\n"
            "Expected location:\n{}".format(config_path)
        )
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        nuke.message("DBFluxFill: Failed to read config.json\n\n{}".format(e))
        return {}


def _is_indie(config):
    """Return True if Nuke Indie mode is enabled in config."""
    return bool(config.get("nuke_indie", False))

def _validate_components(config):
    """
    Check that all model component directories exist.
    Returns a list of error strings, empty if all good.
    """
    components = config.get("components", {})
    component_keys = [
        "transformer", "vae", "text_encoder", "text_encoder_2",
        "tokenizer", "tokenizer_2", "scheduler"
    ]
    missing = []
    for key in component_keys:
        val = components.get(key)
        if not val:
            missing.append("{} - not configured in config.json".format(key))
        elif not os.path.isdir(val):
            missing.append("{} - folder not found: {}".format(key, val))
    return missing

def _validate_environment(config):
    """
    Check that the Python executable exists.
    Returns an error string, or None if all good.
    """
    python_exe = _get_python_exe()
    if not os.path.isfile(python_exe):
        return (
            "Embedded Python not found.\n\n"
            "Expected:\n{}\n\n"
            "Please run setup.bat to set up the environment.".format(python_exe)
        )
    return None

def _build_subprocess_env():
    """
    Build an env dict that prepends the embedded Python directory to PATH.
    This ensures Windows finds python311.dll and all bundled DLLs correctly,
    preventing 0xC0000135 errors regardless of system Python state.
    """
    python_exe = _get_python_exe()
    python_dir = os.path.dirname(python_exe)
    env = os.environ.copy()
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
    env["HF_HUB_OFFLINE"]       = "1"
    env["TRANSFORMERS_OFFLINE"]  = "1"
    return env

# ---------------------------------------------------------------------------
# Daemon management
# ---------------------------------------------------------------------------

def _get_log_path(temp_dir):
    """Return a log file path inside temp_dir for this daemon session."""
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(temp_dir, "DBFluxFill_daemon_{}.log".format(timestamp)).replace("\\", "/")


def _open_log_window(log_path):
    """
    Open a CMD window that tails the log file live using PowerShell.
    Returns the Popen handle or None on failure.
    """
    global _log_window_proc
    cmd = [
        "cmd.exe", "/c", "start",
        "DBFluxFill - Generation Log",
        "powershell.exe", "-Command",
        "while (-not (Test-Path '{0}')) {{ Start-Sleep -Milliseconds 200 }}; Get-Content -Path '{0}' -Wait -Tail 50".format(log_path)
    ]
    try:
        _log_window_proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except Exception as e:
        print("DBFluxFill: Could not open log window: {}".format(e))

def _start_daemon(config, log_path):
    """
    Launch flux_runner.py in daemon mode as a persistent subprocess.
    Starts a reader thread that watches stdout for READY and RESULT signals.
    Daemon writes a live log file to log_path.
    Returns True if the daemon started and signalled READY, False otherwise.
    """
    global _daemon_process, _reader_thread, _daemon_log_path

    _daemon_log_path = log_path

    gizmo_dir   = _get_gizmo_dir()
    python_exe = _get_python_exe()
    runner      = os.path.join(gizmo_dir, "flux_runner.py")
    components  = config.get("components", {})

    cmd = [
        python_exe, runner,
        "--daemon",
        "--log",            log_path,
        "--transformer",    components["transformer"],
        "--vae",            components["vae"],
        "--text_encoder",   components["text_encoder"],
        "--text_encoder_2", components["text_encoder_2"],
        "--tokenizer",      components["tokenizer"],
        "--tokenizer_2",    components["tokenizer_2"],
        "--scheduler",      components["scheduler"],
    ]

    print("DBFluxFill: Starting daemon...")

    _daemon_ready.clear()
    _job_done.clear()

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=0,
            env=_build_subprocess_env(),
        )
    except Exception as e:
        nuke.message("DBFluxFill: Failed to launch daemon.\n\n{}".format(e))
        return False

    _daemon_process = proc

    # Start stdout reader thread (READY and RESULT signals)
    t = threading.Thread(target=_daemon_reader, args=(proc,), name="DBFluxFill-reader")
    t.daemon = True
    t.start()
    _reader_thread = t

    # Start stderr reader thread (model loading progress to script editor)
    t_err = threading.Thread(target=_daemon_stderr_reader, args=(proc,), name="DBFluxFill-stderr")
    t_err.daemon = True
    t_err.start()

    print("DBFluxFill: Waiting for model to load...")
    loaded = _daemon_ready.wait(timeout=300)   # 5 min timeout for model load

    if not loaded:
        nuke.message(
            "DBFluxFill: Daemon did not signal READY within 5 minutes.\n\n"
            "Check the Nuke script editor for errors."
        )
        _kill_daemon_process()
        return False

    print("DBFluxFill: Daemon ready.")
    return True


def _daemon_reader(proc):
    """
    Runs in a background thread. Reads stdout lines from the daemon process.
    Watches for READY and RESULT signals. Prints everything else to the
    Nuke script editor.
    Exits cleanly when the process stdout closes (daemon exited).
    """
    global _job_result

    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line == "DBFLUXFILL_READY":
                _daemon_ready.set()
            elif line.startswith("DBFLUXFILL_RESULT:"):
                try:
                    payload = json.loads(line[len("DBFLUXFILL_RESULT:"):])
                    _job_result = payload
                except Exception as e:
                    _job_result = {"success": False, "error": str(e)}
                _job_done.set()
            else:
                print("DBFluxFill [runner]: {}".format(line))
    except Exception:
        pass
    finally:
        proc.stdout.close()

def _daemon_stderr_reader(proc):
    """
    Runs in a background thread. Reads stderr from the daemon and prints
    it to the Nuke script editor so the user can see model loading progress.
    """
    try:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            print("DBFluxFill [runner]: {}".format(line))
    except Exception:
        pass
    finally:
        proc.stderr.close()

def _kill_daemon_process(wait_timeout=None):
    """Hard-kill the daemon process and reset state."""
    global _daemon_process, _reader_thread
    proc = _daemon_process

    if proc is not None:
        if wait_timeout is not None:
            try:
                proc.wait(timeout=wait_timeout)
                print("DBFluxFill: Daemon exited cleanly.")
            except subprocess.TimeoutExpired:
                print("DBFluxFill: Daemon did not exit in time, killing.")
                proc.kill()
        else:
            try:
                proc.kill()
            except Exception:
                pass

    _daemon_process = None
    _reader_thread  = None
    _daemon_ready.clear()
    _job_done.clear()


def _daemon_is_alive():
    """Return True if the daemon process exists and has not exited."""
    global _daemon_process
    if _daemon_process is None:
        return False
    return _daemon_process.poll() is None


def _send_job(input_path, mask_path, output_path,
              steps, guidance, seed, prompt):
    """
    Send a job to the running daemon via stdin.
    Blocks until the daemon signals RESULT or times out.
    Returns (success, resolved_seed).
    Must be called from a background thread, not the main Nuke thread.
    """
    global _daemon_process, _job_result

    job = {
        "input":    input_path,
        "mask":     mask_path,
        "output":   output_path,
        "steps":    steps,
        "guidance": guidance,
        "seed":     seed,
        "prompt":   prompt,
    }

    line = json.dumps(job) + "\n"

    _job_done.clear()
    _job_result = {}

    print("DBFluxFill: Sending job to daemon stdin...")

    try:
        _daemon_process.stdin.write(line.encode("utf-8"))
        _daemon_process.stdin.flush()
        print("DBFluxFill: Job written to stdin.")
    except Exception as e:
        return False, seed

    # Wait up to 20 minutes for inference to complete
    finished = _job_done.wait(timeout=1200)

    if not finished:
        print("DBFluxFill: Job timed out (20 min) waiting for daemon result.")
        return False, seed

    result = _job_result
    if result.get("success"):
        return True, result.get("seed", seed)
    else:
        print("DBFluxFill [runner]: ERROR - {}".format(result.get("error", "unknown")))
        return False, seed
    

# ---------------------------------------------------------------------------
# TCL expression evaluation
# ---------------------------------------------------------------------------

def _eval_path_knob(node, knob_name):
    """
    Evaluate a path knob that may contain TCL expressions.
    Uses .evaluate() which resolves TCL in node context (Nuke 13.2 compatible).
    Falls back to .getValue() raw string if evaluate fails.
    Normalises slashes for Windows.
    """
    k = node[knob_name]
    try:
        result = k.evaluate()
        if result:
            return result.strip().replace("\\", "/")
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Resolution check
# ---------------------------------------------------------------------------

_MAX_THRESHOLD = 2048
_MIN_THRESHOLD = 512

def _check_resolution(node):
    """
    Check input image resolution.
    Shows a yes/no dialog if resolution exceeds threshold.
    Returns True to proceed, False to abort.
    """
    try:
        # Access the specific node inside the gizmo
        target_node = node.node("CropImgBackup")
        input_node = node.input(0)
        
        if input_node is None:
            nuke.message("DBFluxFill: No image connected to the img input.")
            return False
        if target_node is None:
            nuke.message('DBFluxFill: Internal node "CropImgBackup" not found. The gizmo may be corrupted.')
            return False

        fmt = target_node.format()
        w   = fmt.width()
        h   = fmt.height()

        if w > _MAX_THRESHOLD or h > _MAX_THRESHOLD:
            if not nuke.ask(
                "DBFluxFill: Input resolution is {}x{}.\n\n"
                "Images larger than {}px on either side will take significantly "
                "longer to process and may exceed VRAM limits.\n\n"
                "Continue anyway?".format(w, h, _MAX_THRESHOLD)
            ):
                return False

        # Check for Low Resolution
        if w < _MIN_THRESHOLD or h < _MIN_THRESHOLD:
            if not nuke.ask(
                "DBFluxFill: Input resolution is {}x{}.\n\n"
                "Images smaller than {}px on either side may produce poor results.\n\n"
                "Continue anyway?".format(w, h, _MIN_THRESHOLD)
            ):
                return False

        return True

    except Exception as e:
        nuke.message("DBFluxFill: Could not check input resolution.\n\n{}".format(e))
        return False


# ---------------------------------------------------------------------------
# Write temp files
# ---------------------------------------------------------------------------

def _write_temp_files(node, frame, temp_img_path, temp_mask_path, indie):
    """
    Execute WriteImg and WriteMask nodes at the given frame.
    In Nuke Indie mode, executes one at a time with a step-through dialog.
    Returns True on success, False on failure.
    """
    try:
        write_img  = node.node("WriteImg")
        write_mask = node.node("WriteMask")

        if write_img is None or write_mask is None:
            nuke.message(
                "DBFluxFill: Could not find WriteImg or WriteMask nodes inside the group.\n\n"
                "The group may be corrupted. Try deleting and re-creating the DBFluxFill node."
            )
            return False

        write_img["file"].setValue(temp_img_path)
        write_mask["file"].setValue(temp_mask_path)

        for path in [temp_img_path, temp_mask_path]:
            d = os.path.dirname(path)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)

        if indie:
            # Nuke Indie: execute one node at a time with user confirmation
            if not nuke.ask(
                "DBFluxFill (Indie Mode): Ready to write the image temp file.\n\n"
                "Click OK to continue."
            ):
                return False
            nuke.execute(write_img, frame, frame)

            if not nuke.ask(
                "DBFluxFill (Indie Mode): Image written.\n\n"
                "Ready to write the mask temp file.\n"
                "Click OK to continue."
            ):
                return False
            nuke.execute(write_mask, frame, frame)
        else:
            nuke.execute(write_img,  frame, frame)
            nuke.execute(write_mask, frame, frame)

        return True

    except Exception as e:
        nuke.message(
            "DBFluxFill: Failed to write temp files.\n\n"
            "{}\n\n"
            "Check that the Temp Directory path is valid and writable.".format(e)
        )
        return False


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_flux_oneshot(config, temp_img, temp_mask, output_path,
                      steps, guidance, seed, prompt, components):
    """
    Single-shot mode: launch flux_runner.py, wait for it to finish, exit.
    Used by plain Generate button (no stay-loaded).
    Returns (success, resolved_seed).
    """
    gizmo_dir   = _get_gizmo_dir()
    python_exe = _get_python_exe()
    runner      = os.path.join(gizmo_dir, "flux_runner.py")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir)
        except Exception as e:
            nuke.message("DBFluxFill: Could not create output directory.\n\n{}".format(e))
            return False, -1

    cmd = [
        python_exe, runner,
        "--input",          temp_img,
        "--mask",           temp_mask,
        "--output",         output_path,
        "--transformer",    components["transformer"],
        "--vae",            components["vae"],
        "--text_encoder",   components["text_encoder"],
        "--text_encoder_2", components["text_encoder_2"],
        "--tokenizer",      components["tokenizer"],
        "--tokenizer_2",    components["tokenizer_2"],
        "--scheduler",      components["scheduler"],
        "--steps",          str(steps),
        "--guidance",       str(guidance),
        "--seed",           str(seed),
        "--prompt",         prompt if prompt else " ",
    ]

    print("DBFluxFill: Launching one-shot runner...")

    try:
        nuke.message(
            "DBFluxFill: FLUX inference is starting.\n\n"
            "A CMD window will open showing progress.\n"
            "Nuke will be unresponsive until generation completes.\n\n"
            "Click OK to start."
        )

        process = subprocess.Popen(
            [python_exe, runner] + cmd[2:],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=_build_subprocess_env(),
        )
        process.wait()

        resolved_seed = seed
        seed_file = output_path + ".seed"
        try:
            if os.path.isfile(seed_file):
                with open(seed_file, "r") as f:
                    resolved_seed = int(f.read().strip())
                os.remove(seed_file)
        except Exception:
            pass

        if process.returncode != 0:
            nuke.message(
                "DBFluxFill: Runner failed (exit code {}).\n\n"
                "Check the CMD window for details.".format(process.returncode)
            )
            return False, seed

        if not os.path.isfile(output_path):
            nuke.message(
                "DBFluxFill: Runner reported success but output file not found.\n\n"
                "Expected:\n{}".format(output_path)
            )
            return False, seed

        return True, resolved_seed

    except Exception as e:
        nuke.message("DBFluxFill: Failed to launch runner.\n\n{}".format(e))
        traceback.print_exc()
        return False, seed


def _run_flux_daemon(node, config, temp_dir, temp_img, temp_mask, output_path,
                     steps, guidance, seed, prompt):
    """
    Daemon mode: ensure daemon is running, send job, wait for result.
    Opens a live log window on first launch only, reuses it for subsequent jobs.
    Returns (success, resolved_seed).
    """
    global _daemon_process, _daemon_log_path

    with _daemon_lock:
        if not _daemon_is_alive():
            log_path = _get_log_path(temp_dir)
            _daemon_log_path = log_path

            try:
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.isdir(log_dir):
                    os.makedirs(log_dir)
                with open(log_path, "w") as f:
                    f.write("DBFluxFill daemon log\n")
                    f.write("=" * 40 + "\n")
            except Exception as e:
                print("DBFluxFill: Could not create log file: {}".format(e))
                log_path = None
                _daemon_log_path = None

            if log_path:
                _open_log_window(log_path)

            if not _start_daemon(config, log_path):
                return False, seed
            node["daemon_running"].setValue(True)
        else:
            print("DBFluxFill: Daemon already running, reusing loaded model.")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir)
        except Exception as e:
            nuke.message("DBFluxFill: Could not create output directory.\n\n{}".format(e))
            return False, seed

    result_holder = [False, seed]
    done_event    = threading.Event()

    def worker():
        success, resolved_seed = _send_job(
            temp_img, temp_mask, output_path,
            steps, guidance, seed, prompt
        )
        result_holder[0] = success
        result_holder[1] = resolved_seed
        done_event.set()

    t = threading.Thread(target=worker, name="DBFluxFill-job")
    t.daemon = True
    t.start()

    print("DBFluxFill: Job sent to daemon. Check the log window for progress.")

    # _send_job blocks for up to 20 minutes waiting for DBFLUXFILL_RESULT.
    # t.join() returns immediately once that timeout expires.
    t.join()

    return result_holder[0], result_holder[1]

# ---------------------------------------------------------------------------
# Drop result nodes
# ---------------------------------------------------------------------------

def _drop_result_nodes(node, output_path, frame, indie, seed, steps, guidance, prompt):
    """
    Drop Read + Reformat + Transform nodes into the graph after generation.
    Nodes are placed to the right of the gizmo, not connected to anything.
    """
    if indie:
        if not nuke.ask(
            "DBFluxFill (Indie Mode): Generation complete.\n\n"
            "Ready to add the result node to the graph.\n"
            "Click OK to continue."
        ):
            return

    try:
        gizmo_x = node.xpos()
        gizmo_y = node.ypos()
        drop_x  = gizmo_x
        drop_y  = gizmo_y + 200

        # Get crop offset BEFORE entering root context
        tx, ty = 0, 0
        was_scaled_up = False
        scaled_w = 0
        scaled_h = 0
        try:
            if node["crop_to_mask"].value():
                with node:
                    minsize_node = node.node("ReformatImgMinSize")
                    was_scaled_up = minsize_node and not minsize_node["disable"].value()
                    scaled_w = int(minsize_node["box_width"].value()) if was_scaled_up else 0
                    scaled_h = int(minsize_node["box_height"].value()) if was_scaled_up else 0
                    crop_node = node.node("CropImgToBBox")
                    if crop_node:
                        tx = crop_node.knob("box").value()[0]
                        ty = crop_node.knob("box").value()[1]
                        print("DBFluxFill: bbox inside group tx={} ty={}".format(tx, ty))
        except Exception as e:
            print("DBFluxFill: bbox fetch failed: {}".format(e))

        print("DBFluxFill: was_scaled_up={} scaled_w={} scaled_h={}".format(was_scaled_up, scaled_w, scaled_h))
        
        with nuke.root():
            # Read node
            read = nuke.nodes.Read()
            read["file"].setValue(output_path.replace("\\", "/"))
            read["colorspace"].setValue("sRGB")
            read["first"].setValue(frame)
            read["last"].setValue(frame)
            read["origfirst"].setValue(frame)
            read["origlast"].setValue(frame)
            read.addKnob(nuke.Tab_Knob("fluxfill_tab", "DBFluxFill Settings"))
            seed_knob = nuke.String_Knob("fluxfill_seed", "Seed")
            seed_knob.setValue(str(seed))
            seed_knob.setFlag(nuke.READ_ONLY)
            read.addKnob(seed_knob)
            steps_knob = nuke.String_Knob("fluxfill_steps", "Steps")
            steps_knob.setValue(str(steps))
            steps_knob.setFlag(nuke.READ_ONLY)
            read.addKnob(steps_knob)
            guidance_knob = nuke.String_Knob("fluxfill_guidance", "Guidance")
            guidance_knob.setValue(str(guidance))
            guidance_knob.setFlag(nuke.READ_ONLY)
            read.addKnob(guidance_knob)
            prompt_knob = nuke.String_Knob("fluxfill_prompt", "Prompt")
            prompt_knob.setValue(prompt if prompt else "")
            prompt_knob.setFlag(nuke.READ_ONLY)
            read.addKnob(prompt_knob)
            read.addKnob(nuke.Text_Knob("fluxfill_divider","",""))
            info_knob = nuke.Text_Knob("fluxfill_info", "", "Above are the settings used to generate this image.\nDeleting this node will delete this tab, so make sure you copy\nany settings you want to keep before deleting generated nodes.")
            read.addKnob(info_knob)
            read.setXYpos(drop_x, drop_y)

            # Reformat to match original input canvas
            reformat = None
            try:
                input_fmt = node.input(0).format()
                w = input_fmt.width()
                h = input_fmt.height()

                last_node = read
                if was_scaled_up:
                    crop = nuke.nodes.Crop()
                    crop["box"].setValue([0, 0, scaled_w, scaled_h])
                    crop["reformat"].setValue(True)
                    crop["crop"].setValue(False)
                    crop.setInput(0, read)
                    crop.setXYpos(drop_x, drop_y + 80)
                    last_node = crop

                reformat = nuke.nodes.Reformat()
                reformat["type"].setValue("to box")
                reformat["box_width"].setValue(w)
                reformat["box_height"].setValue(h)
                reformat["resize"].setValue("distort" if was_scaled_up else "none")
                reformat["center"].setValue(False)
                reformat["pbb"].setValue(True)
                reformat.setInput(0, last_node)
                reformat.setXYpos(drop_x, drop_y + (160 if was_scaled_up else 80))
            except Exception:
                pass

            # Transform for crop offset
            print("DBFluxFill: tx={} ty={} reformat={}".format(tx, ty, reformat))
            if (tx != 0 or ty != 0) and reformat is not None:
                print("DBFluxFill: Dropping transform")
                transform = nuke.nodes.Transform()
                transform["translate"].setValue([tx, ty])
                transform["center"].setValue([0, 0])
                transform.setInput(0, reformat)
                transform.setXYpos(drop_x, drop_y + (240 if was_scaled_up else 160))
            else:
                print("DBFluxFill: Transform skipped - condition not met")

        print("DBFluxFill: Result nodes dropped.")
        read.setSelected(True)
        nuke.zoom(1, [read.xpos(), read.ypos()])

    except Exception as e:
        print("DBFluxFill: Warning - could not drop result nodes: {}".format(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Generate callbacks
# ---------------------------------------------------------------------------

def _prepare_generate(node):
    """
    Shared setup for both generate paths.
    Returns a dict of everything needed, or None to abort.
    """
    config = _load_config()
    if not config:
        return None

    indie = _is_indie(config)

    component_errors = _validate_components(config)
    if component_errors:
        nuke.message(
            "DBFluxFill: One or more model components are missing or not configured.\n\n"
            "{}\n\n"
            "Please check your config.json and run setup.bat if needed.".format(
                "\n".join(component_errors))
        )
        return None
    
    env_error = _validate_environment(config)
    if env_error:
        nuke.message("DBFluxFill: {}".format(env_error))
        return None

    if not _check_resolution(node):
        return None

    frame    = int(node["framehold_cntrl"].getValue())
    steps    = int(node["steps"].getValue())
    guidance = float(node["guidance"].getValue())
    seed     = int(node["seed"].getValue())
    prompt   = node["prompt"].getValue().strip()

    temp_dir    = _eval_path_knob(node, "temp_dir")
    output_dir  = _eval_path_knob(node, "output_dir")
    output_name = _eval_path_knob(node, "output_name")

    script_dir = os.path.dirname(nuke.root().name())
    if not temp_dir or not os.path.isabs(temp_dir):
        temp_dir = script_dir
    if not output_dir or not os.path.isabs(output_dir):
        output_dir = script_dir

    base_name      = output_name if output_name else "fluxfill"
    temp_img_path  = os.path.join(temp_dir,  "{}-tempImg.png".format(base_name)).replace("\\", "/")
    temp_mask_path = os.path.join(temp_dir,  "{}-tempMask.png".format(base_name)).replace("\\", "/")
    output_path    = os.path.join(output_dir, "{}.png".format(base_name)).replace("\\", "/")

    print("DBFluxFill: Generate started - frame {}".format(frame))
    print("DBFluxFill: Temp img  -> {}".format(temp_img_path))
    print("DBFluxFill: Temp mask -> {}".format(temp_mask_path))
    print("DBFluxFill: Output    -> {}".format(output_path))

    if not _write_temp_files(node, frame, temp_img_path, temp_mask_path, indie):
        return None

    return {
        "config":          config,
        "indie":           indie,
        "frame":           frame,
        "steps":           steps,
        "guidance":        guidance,
        "seed":            seed,
        "prompt":          prompt,
        "temp_dir":        temp_dir,
        "temp_img_path":   temp_img_path,
        "temp_mask_path":  temp_mask_path,
        "output_path":     output_path,
        "components":      config.get("components", {}),
    }


def on_generate(node):
    """Generate button - one-shot, daemon exits after completion."""
    ctx = _prepare_generate(node)
    if ctx is None:
        return

    # If daemon is alive from a previous stay-loaded run, reject cleanly
    if _daemon_is_alive():
        nuke.message(
            "DBFluxFill: The model is currently loaded in VRAM.\n\n"
            "Use 'Generate (Stay Loaded)' to run another job with the "
            "loaded model, or press 'Unload Model' first."
        )
        return

    success, resolved_seed = _run_flux_oneshot(
        config      = ctx["config"],
        temp_img    = ctx["temp_img_path"],
        temp_mask   = ctx["temp_mask_path"],
        output_path = ctx["output_path"],
        steps       = ctx["steps"],
        guidance    = ctx["guidance"],
        seed        = ctx["seed"],
        prompt      = ctx["prompt"],
        components  = ctx["components"],
    )

    if not success:
        return

    _drop_result_nodes(
        node, ctx["output_path"], ctx["frame"], ctx["indie"],
        resolved_seed, ctx["steps"], ctx["guidance"], ctx["prompt"]
    )

    nuke.message(
        "DBFluxFill: Generation complete.\n\n"
        "Output saved to:\n{}".format(ctx["output_path"])
    )


def on_generate_stay(node):
    """Generate (Stay Loaded) button - launches or reuses daemon, keeps model in VRAM."""
    ctx = _prepare_generate(node)
    if ctx is None:
        return

    # Reject if a job is already in flight
    if _daemon_is_alive() and not _job_done.is_set():
        nuke.message(
            "DBFluxFill: A generation is already in progress.\n\n"
            "Please wait for it to finish."
        )
        return

    nuke.message(
        "DBFluxFill: Generation starting.\n\n"
        "Nuke will be unresponsive during generation.\n"
        "A log window will open so you can monitor progress.\n\n"
        "Click OK to start."
    )

    success, resolved_seed = _run_flux_daemon(
        node        = node,
        config      = ctx["config"],
        temp_dir    = ctx["temp_dir"],
        temp_img    = ctx["temp_img_path"],
        temp_mask   = ctx["temp_mask_path"],
        output_path = ctx["output_path"],
        steps       = ctx["steps"],
        guidance    = ctx["guidance"],
        seed        = ctx["seed"],
        prompt      = ctx["prompt"],
    )

    if not success:
        nuke.message(
            "DBFluxFill: Generation failed.\n\n"
            "Check the Nuke script editor for details."
        )
        return

    _drop_result_nodes(
        node, ctx["output_path"], ctx["frame"], ctx["indie"],
        resolved_seed, ctx["steps"], ctx["guidance"], ctx["prompt"]
    )

    nuke.message(
        "DBFluxFill: Generation complete.\n\n"
        "Output saved to:\n{}\n\n"
        "Model remains loaded. Press 'Generate (Stay Loaded)' to generate "
        "again without reloading, or 'Unload Model' to free VRAM.".format(
            ctx["output_path"])
    )


def on_unload_model(node):
    """Unload Model button - sends shutdown to daemon and waits for clean exit."""
    if not _daemon_is_alive():
        nuke.message("DBFluxFill: No model is currently loaded.")
        return

    if not nuke.ask(
        "DBFluxFill: Unload the model from VRAM?\n\n"
        "The daemon process will be shut down cleanly."
    ):
        return

    global _daemon_process
    proc = _daemon_process

    try:
        shutdown_msg = json.dumps({"shutdown": True}) + "\n"
        proc.stdin.write(shutdown_msg.encode("utf-8"))
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass

    _kill_daemon_process(wait_timeout=30)
    node["daemon_running"].setValue(False)

    global _log_window_proc
    if _log_window_proc is not None:
        try:
            _log_window_proc.kill()
        except Exception:
            pass
        _log_window_proc = None

    if _daemon_log_path and os.path.isfile(_daemon_log_path):
        try:
            os.remove(_daemon_log_path)
            print("DBFluxFill: Daemon log deleted.")
        except Exception as e:
            print("DBFluxFill: Could not delete log file: {}".format(e))

    print("DBFluxFill: Model unloaded.")
    nuke.message("DBFluxFill: Model unloaded from VRAM.")

# ---------------------------------------------------------------------------
# Knob change callback
# ---------------------------------------------------------------------------

def on_knob_changed(node, knob):
    """Update path preview labels when source knobs change or panel opens."""
    watched = {
        "temp_dir":    "temp_dir_prev",
        "output_dir":  "output_dir_prev",
        "output_name": "output_name_prev",
    }
    name = knob.name()

    # Refresh all previews when the properties panel opens
    if name == "showPanel":
        for src, preview in watched.items():
            value = _eval_path_knob(node, src)
            if src == "output_name":
                value = value + ".png"
            node[preview].setValue(value)
        return

    if name in watched:
        evaluated = _eval_path_knob(node, name)
        if name == "output_name":
            evaluated = evaluated + ".png"
        node[watched[name]].setValue(evaluated)


# ---------------------------------------------------------------------------
# Frame hold callback on node creation
# ---------------------------------------------------------------------------

def on_create(node):
    try:
        node["framehold_cntrl"].setValue(int(nuke.frame()))
    except Exception as e:
        print("DBFluxFill: on_create - could not set framehold_cntrl: {}".format(e))
    try:
        if not node["paths_initialised"].value():
            config = _load_config()
            for knob_name in ("temp_dir", "output_dir", "output_name"):
                val = config.get(knob_name)
                if val:
                    node[knob_name].setValue(val)
            node["paths_initialised"].setValue(True)
    except Exception as e:
        print("DBFluxFill: on_create - could not initialize paths: {}".format(e))
    try:
        on_knob_changed(node, node["showPanel"])
    except Exception as e:
        print("DBFluxFill: on_create - could not refresh path previews: {}".format(e))
    try:
        if nuke.NUKE_VERSION_MAJOR >= 15:
            node["disable_group_view"].setValue(True)
    except Exception as e:
        print("DBFluxFill: on_create - could not disable group view: {}".format(e))
