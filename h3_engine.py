"""Syok Swap H3 engine: Openjutsu's 'Simple' swap and 'Check mask' workflows run headlessly in ComfyUI (pure Python here).

No Modal, no GPU, no network in this file, so every rule is unit-testable. h3_app.py is the Modal side (image, Volumes,
the ComfyUI process).

Workflows (copied from D3voz/openjutsu, MIT, see workflows/openjutsu/LICENSE and THIRD-PARTY.md):
    workflows/openjutsu/simple.api.json   the swap: SAM 3.1 tracks the main person, MiniMax H3 redraws them, clip by clip,
                                          through the Context Loop (checkpoints per clip, so a rerun resumes)
    workflows/openjutsu/mask.api.json     the mask check: same SAM + mask settings over the section, cyan/orange preview MP4

The patching below is a Python port of openjutsu lib/simple.ts (pruneSimple, pruneMask, setSection, setResolution,
cropSource, dropNoLipSync, setMotionSize, suffixRunName) and lib/clips.ts (clip plan estimate), same node ids, same rules.
lib/model.ts is NOT ported: with the default model files it changes nothing in the export (the export already names them).

A job is a folder under a root directory (on Modal: the `syok-h3-jobs` Volume), with the job folder
helpers (texture_service.new_job / read_meta / write_meta / set_stage). meta.json carries two H3 stages:

    check   "idle" -> "queued" -> "running" -> "ready" | "error"         small GPU: SAM only, writes check.mp4
    swap    "idle" -> "queued" -> "generating" -> "done" | "error"      big GPU, the paid step: writes h3.mp4

Resume: ComfyUI's output folder for a job is <job>/comfy_out on the Volume. The Context Loop saves each finished clip and
a checkpoint there, keyed by the Swap Plan's run name, which depends on the inputs. Starting the swap again on the SAME job
after an error (same files, same settings) therefore continues from the last finished clip (openjutsu README: "Running the
same inputs again resumes from the last finished clip"). The settings are frozen at upload, so a resume cannot change them.

Files:  raw.mp4 (source video)  ref.<ext> (new person)  comfy_in/ (ComfyUI input folder)  comfy_out/ (outputs + checkpoints)
        check.mp4 (mask check preview)  h3.mp4 (swap result)  graph_check.json / graph_swap.json (the exact graph submitted)
"""
import copy
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import texture_service as tsvc
from texture_service import JobError

HERE = Path(__file__).resolve().parent
WORKFLOW_DIR = HERE / "workflows" / "openjutsu"

# ---------------------------------------------------------------- constants from openjutsu lib/simple.ts

PRESETS = ("480p", "576p", "736p", "768p", "1080p")
CUSTOM_SIZE = {"min": 256, "max": 2048, "step": 32}
CLIP_SECONDS = {"min": 2.5, "max": 15.0, "default": 6.0}
MAX_SEED = 0xFFFFFFFF
MARGINS = {"tight": 20, "normal": 40, "loose": 60}        # GrowMaskWithBlur expand, px at SAM's 1008 px canvas
MOTION_SIZES = (512, 768, 1376)
MIN_SECTION_SECONDS = 2.0
FPS = 24                                                 # every export loads the video at force_rate 24
DEFAULT_TARGET = "person"

# Our own limits (not from openjutsu): 30 s per job, to keep one run's cost bounded.
MAX_SECTION_SECONDS = 30.0
DEFAULT_RESOLUTION = "480p"   # openjutsu's page defaults to 736p; 480p is the cheapest preset

# Node ids in workflows/openjutsu/simple.api.json (openjutsu NODES).
NODES = {"picture": "2", "video": "5", "plan": "11", "target": "24", "loopStart": "28", "peopleMask": "39",
         "growMask": "42", "loopEnd": "66", "final": "67", "hat": ["47", "48", "49", "50", "51", "52"]}
# Node ids in workflows/openjutsu/mask.api.json (openjutsu MASK_NODES), plus the preview writer.
MASK_NODES = {"video": "1", "target": "3", "peopleMask": "7", "growMask": "9", "hat": ["10", "11", "12", "13", "19", "20"],
              "preview": "18"}

# The No lip-sync nodes, matched by title like openjutsu's MUTE / NO_LIP_SYNC regexes.
MUTE = re.compile(r"^Mute the (clip|reference) audio H3 hears$")
NO_LIP_SYNC = re.compile(r"^(No lip-sync in the prompt|SAM: faces|SAM 3\.1 track faces|Faces \(shown on the mannequins?\))$")

# Model files the two workflows load, by ComfyUI models/ subfolder (openjutsu README "Models" table, Simple + Check mask only).
# repo_id / path_in_repo: where each file is downloaded from on Hugging Face. Verified on 21 Nov 2026 by listing each repo's
# files (names and sizes match the openjutsu README table); the Comfy-Org repos are the official ComfyUI repackages.
# The files themselves were not downloaded or hash-checked here. fetch still takes --sources to override any of them.
MODEL_FILES = [
    {"folder": "diffusion_models", "file": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "for": ("swap",)},
    {"folder": "loras", "file": "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors", "for": ("swap",)},
    {"folder": "text_encoders", "file": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors", "for": ("swap",)},
    {"folder": "text_encoders", "file": "qwen3vl_8b_int8_convrot.safetensors", "for": ("swap",)},
    {"folder": "vae", "file": "minimax_h3_video_vae_int8_convrot.safetensors", "for": ("swap",)},
    {"folder": "vae", "file": "minimax_h3_audio_vae_fp32.safetensors", "for": ("swap",)},
    {"folder": "checkpoints", "file": "sam3.1_multiplex_fp16.safetensors", "for": ("swap", "check")},
]
MODEL_SOURCES = {
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": {"repo_id": "Comfy-Org/MiniMax-H3", "path_in_repo": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "revision": None},  # 21.0 GB
    "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors": {"repo_id": "Kijai/MiniMax-H3-experimental", "path_in_repo": "loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors", "revision": None},  # 366.6 MB
    "qwen3vl_32b_minimax_h3_int4_convrot.safetensors": {"repo_id": "Merserk/MiniMax-H3-INT4-ConvRot", "path_in_repo": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors", "revision": None},  # 15.0 GB
    "qwen3vl_8b_int8_convrot.safetensors": {"repo_id": "Comfy-Org/Qwen3-VL", "path_in_repo": "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "revision": None},  # 9.4 GB
    "minimax_h3_video_vae_int8_convrot.safetensors": {"repo_id": "Comfy-Org/MiniMax-H3", "path_in_repo": "vae/minimax_h3_video_vae_int8_convrot.safetensors", "revision": None},  # 2.8 GB
    "minimax_h3_audio_vae_fp32.safetensors": {"repo_id": "Comfy-Org/MiniMax-H3", "path_in_repo": "vae/minimax_h3_audio_vae_fp32.safetensors", "revision": None},  # 605 MB
    "sam3.1_multiplex_fp16.safetensors": {"repo_id": "Comfy-Org/sam3.1", "path_in_repo": "checkpoints/sam3.1_multiplex_fp16.safetensors", "revision": None},  # 1.7 GB
}


# ---------------------------------------------------------------- graph helpers (port of lib/simple.ts)

def load_workflow(name):
    """The bundled API export: "simple" or "mask"."""
    if name not in ("simple", "mask"):
        raise ValueError("Workflow must be simple or mask.")
    return json.loads((WORKFLOW_DIR / f"{name}.api.json").read_text())


def is_link(value):
    """[node_id, output_index], as in the API export (isLink)."""
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def check_links(graph):
    """Every link must point at a node that still exists (the check maskSettings ends with)."""
    for nid, node in graph.items():
        for name, value in node["inputs"].items():
            if is_link(value) and value[0] not in graph:
                raise ValueError(f"Node {nid}.{name} still links to removed node {value[0]}.")


def canvas_size(width, height, preset):
    """Canvas the Swap Plan makes for a source of this size (canvasSize). Math.round == floor(x + 0.5) for these values."""
    scale = int(preset[:-1]) / min(width, height)
    return [max(32, math.floor(side * scale / 32 + 0.5) * 32) for side in (width, height)]


def crop_source(graph, video, size):
    """Scale and centre-crop the source frames before anything reads them (cropSource)."""
    for node in graph.values():
        for name, value in list(node["inputs"].items()):
            if is_link(value) and value[0] == video and value[1] == 0:
                node["inputs"][name] = ["custom_size", 0]
    graph["custom_size"] = {"class_type": "ImageScale",
                            "inputs": {"image": [video, 0], "upscale_method": "bilinear", "width": size["width"],
                                       "height": size["height"], "crop": "center"},
                            "_meta": {"title": "Custom size (centre crop)"}}


def set_resolution(graph, plan, video, resolution):
    """A preset name is the short edge (keeps the source shape); a {"width", "height"} dict is the exact canvas (setResolution).
    The custom size needs the Swap Plan's `short_edge` input (a recent comfyui-h3-oneclick-swap)."""
    if isinstance(resolution, str):
        graph[plan]["inputs"]["resolution"] = resolution
        return
    crop_source(graph, video, resolution)
    graph[plan]["inputs"]["short_edge"] = min(resolution["width"], resolution["height"])


def set_section(graph, video, start, length):
    """Part of the source in seconds; length 0 runs to the end. VHS cuts the soundtrack to the same part (setSection)."""
    graph[video]["inputs"]["skip_first_frames"] = round_half_up(start * FPS)
    graph[video]["inputs"]["frame_load_cap"] = round_half_up(length * FPS)


def round_half_up(x):
    """JavaScript's Math.round for the non-negative values used here (Python's round() is banker's rounding)."""
    return int(math.floor(x + 0.5))


def suffix_run_name(graph, nid, suffix, title):
    """Settings the Swap Plan's run name does not know get their own run, so a rerun never resumes from clips made without
    them (suffixRunName)."""
    plans = [n for n in graph.values() if n["class_type"] == "MiniMaxH3ChainPlanModern"]
    if len(plans) != 1:
        raise ValueError(f"Expected one Production Plan node in the workflow, found {len(plans)}.")
    graph[nid] = {"class_type": "StringConcatenate",
                  "inputs": {"string_a": plans[0]["inputs"]["run_name"], "string_b": suffix, "delimiter": ""},
                  "_meta": {"title": title}}
    plans[0]["inputs"]["run_name"] = [nid, 0]


def set_motion_size(graph, size):
    """Long edge of the grey-mannequin motion reference (setMotionSize); non-default sizes get their own run name."""
    nodes = [n for n in graph.values() if n["class_type"] == "ImageScaleToMaxDimension"
             and re.search("motion reference", (n.get("_meta") or {}).get("title", ""), re.I)]
    if not nodes:
        raise ValueError("Expected a motion reference resize in the workflow.")
    for n in nodes:
        n["inputs"]["largest_size"] = size
    if size == MOTION_SIZES[0]:
        return
    suffix_run_name(graph, "motion_run_name", f"_motion{size}", "Run name for this motion reference size")


def drop_no_lip_sync(graph):
    """No lip-sync off: the nodes go, H3 hears the soundtrack again and the mannequins get no faces (dropNoLipSync)."""
    through = {}
    for nid, node in graph.items():
        title = (node.get("_meta") or {}).get("title", "")
        if MUTE.match(title):
            through[nid] = node["inputs"].get("audio")
        elif NO_LIP_SYNC.match(title):
            through[nid] = None
    for nid in through:
        del graph[nid]
    for nid, node in graph.items():
        for name, value in list(node["inputs"].items()):
            if not is_link(value) or value[0] not in through:
                continue
            if through[value[0]]:
                node["inputs"][name] = through[value[0]]
            elif name in ("no_lip_sync", "faces", "face_size"):
                del node["inputs"][name]
            else:
                raise ValueError(f"Node {nid}.{name} still links to removed node {value[0]}.")


def _mask_settings(source, hat_fix, margin, target, nodes):
    graph = copy.deepcopy(source)
    graph[nodes["target"]]["inputs"]["text"] = target
    graph[nodes["growMask"]]["inputs"]["expand"] = MARGINS[margin]
    if not hat_fix:
        for nid in nodes["hat"]:
            del graph[nid]
        graph[nodes["growMask"]]["inputs"]["mask"] = [nodes["peopleMask"], 0]
    check_links(graph)
    return graph


def prune_simple(source, s):
    """The swap graph for settings `s` (see normalize_settings). The source graph is not modified (pruneSimple)."""
    graph = _mask_settings(source, s["hat_fix"], s["margin"], s["target"], NODES)
    if not s["no_lip_sync"]:
        drop_no_lip_sync(graph)
    set_motion_size(graph, s["motion"])
    set_resolution(graph, NODES["plan"], NODES["video"], s["resolution"])
    plan = graph[NODES["plan"]]["inputs"]
    plan["seed"] = s["seed"]
    plan["clip_seconds"] = s["clip_seconds"]
    plan["roles"] = re.sub(r"\s+", " ", s["role"]).strip()
    check_links(graph)
    return graph


def prune_mask(source, hat_fix, margin, target):
    """The mask-check graph (pruneMask)."""
    return _mask_settings(source, hat_fix, margin, target, MASK_NODES)


def bind_inputs(graph, video_node, video_name, picture_node=None, picture_name=None):
    """Put the uploaded files' names (inside ComfyUI's input folder) on the loaders, as lib/comfy.ts setInput does."""
    if graph[video_node]["class_type"] != "VHS_LoadVideo":
        raise ValueError("The video node is not a VHS_LoadVideo.")
    graph[video_node]["inputs"]["video"] = video_name
    if picture_node is not None:
        if graph[picture_node]["class_type"] != "LoadImage":
            raise ValueError("The picture node is not a LoadImage.")
        graph[picture_node]["inputs"]["image"] = picture_name
    return graph


def swap_graph(s, video_name, picture_name, source=None):
    """Everything the swap submits: pruned, section set, files bound."""
    g = prune_simple(source or load_workflow("simple"), s)
    set_section(g, NODES["video"], s["start"], s["length"])
    return bind_inputs(g, NODES["video"], video_name, NODES["picture"], picture_name)


def check_graph(s, video_name, source=None):
    """Everything the mask check submits. The check uses the same section as the swap (openjutsu: "Check mask uses it too")."""
    g = prune_mask(source or load_workflow("mask"), s["hat_fix"], s["margin"], s["target"])
    set_section(g, MASK_NODES["video"], s["start"], s["length"])
    return bind_inputs(g, MASK_NODES["video"], video_name)


# ---------------------------------------------------------------- clip plan estimate (port of lib/clips.ts)

OVERLAP = 39


def _grid_floor(frames):
    return (frames - 5) // 17 * 17 + 5


def _clip_plan(duration, clip_seconds):
    try:
        total = math.floor(duration * FPS)
        longest = _grid_floor(math.floor(clip_seconds * FPS))
    except (OverflowError, ValueError):
        return []
    if total < 5 or longest <= OVERLAP:
        return []
    clips, end, context = [], 0, 0
    while True:
        start = end - context
        length = _grid_floor(min(longest, total - start))
        if clips and length - context < 17:
            return clips
        clips.append((length, context))
        end = start + length
        context = min(OVERLAP, _grid_floor(length - context))


def clip_count(duration, clip_seconds):
    """About how many H3 clips a run takes (exact for a video without hard cuts)."""
    return len(_clip_plan(duration, clip_seconds))


def clip_starts(duration, clip_seconds):
    starts, end = [], 0
    for length, context in _clip_plan(duration, clip_seconds):
        starts.append(end - context)
        end = starts[-1] + length
    return starts or [0]


# ---------------------------------------------------------------- settings

def _number(v, name, lo=None, hi=None):
    try:
        n = float(v)
    except (TypeError, ValueError):
        raise JobError(f"{name} must be a number.")
    if n != n or n in (float("inf"), float("-inf")) or (lo is not None and n < lo) or (hi is not None and n > hi):
        raise JobError(f"{name} must be between {lo} and {hi}." if hi is not None else f"{name} must be {lo} or more.")
    return n


def parse_resolution(value):
    """"480p" (a preset) or "WIDTHxHEIGHT" / {"width", "height"} (custom, 256 to 2048 px a side, rounded to 32)."""
    if value is None or value == "":
        return DEFAULT_RESOLUTION
    if isinstance(value, str) and value in PRESETS:
        return value
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if not m:
            raise JobError("Resolution must be " + ", ".join(PRESETS) + " or WIDTHxHEIGHT.")
        value = {"width": int(m.group(1)), "height": int(m.group(2))}
    if not isinstance(value, dict):
        raise JobError("Resolution must be a preset or a width and height.")
    out = {}
    for side in ("width", "height"):
        n = _number(value.get(side), side.capitalize(), CUSTOM_SIZE["min"], CUSTOM_SIZE["max"])
        out[side] = int(round_half_up(n / CUSTOM_SIZE["step"]) * CUSTOM_SIZE["step"])
    return out


def normalize_settings(margin="normal", start=0.0, length=0.0, seed=0, resolution=DEFAULT_RESOLUTION, role="",
                       target=DEFAULT_TARGET, hat_fix=False, no_lip_sync=False, motion=512,
                       clip_seconds=CLIP_SECONDS["default"]):
    """Check and normalise every setting BEFORE anything is stored. Start/length are checked against the video later."""
    margin = (margin or "normal").strip().lower()
    if margin not in MARGINS:
        raise JobError("Margin must be tight, normal or loose.")
    seed_f = _number(seed if seed not in (None, "") else 0, "Seed", 0, MAX_SEED)
    if seed_f != int(seed_f):
        raise JobError("Seed must be a whole number.")
    motion_f = _number(motion, "Motion reference size")
    if motion_f not in MOTION_SIZES:
        raise JobError("Motion reference size must be 512, 768 or 1376.")
    target = re.sub(r"\s+", " ", target or "").strip() or DEFAULT_TARGET
    if len(target) > 80:
        raise JobError("What to replace must be a short description.")
    role = role or ""
    if len(role) > 200:
        raise JobError("Who they replace must be short.")
    return {"margin": margin, "start": _number(start or 0, "Start", 0, 86400), "length": _number(length or 0, "Length", 0, 86400),
            "seed": int(seed_f), "resolution": parse_resolution(resolution), "role": role, "target": target,
            "hat_fix": bool(hat_fix), "no_lip_sync": bool(no_lip_sync), "motion": int(motion_f),
            "clip_seconds": _number(clip_seconds, "Clip length", CLIP_SECONDS["min"], CLIP_SECONDS["max"])}


def plan_section(duration, start, length):
    """Clamp a (start, length) request to the video. length 0 = to the end. Returns (start, length) in seconds."""
    if not duration or duration <= 0:
        raise JobError("That video has no duration.")
    start = float(start)
    if start > duration - MIN_SECTION_SECONDS:
        raise JobError(f"The part must start at least {MIN_SECTION_SECONDS:g} seconds before the end of the video.")
    end = duration if not length else min(duration, start + float(length))
    length = end - start
    if length < MIN_SECTION_SECONDS:
        raise JobError(f"The part must be at least {MIN_SECTION_SECONDS:g} seconds long.")
    if length > MAX_SECTION_SECONDS:
        raise JobError(f"The part is {length:.1f} s; the limit is {MAX_SECTION_SECONDS:.0f} s per job.")
    return round(start, 3), round(length, 3)


# ---------------------------------------------------------------- job folder (meta.json, texture_service helpers)

KIND = "h3"
VIDEO_NAME = "raw.mp4"
IN_DIR, OUT_DIR = "comfy_in", "comfy_out"
CHECK_FILE, SWAP_FILE = "check.mp4", "h3.mp4"
ACTIVE = {"check": ("queued", "running"), "swap": ("queued", "generating")}
STUCK_AFTER_SECONDS = {"check": 3600 + 300, "swap": 4 * 3600 + 300}       # the Modal timeouts in h3_app.py plus a margin
STUCK_MESSAGE = {
    "check": "The mask check stopped without finishing (Modal's time limit, a restart, or no credit left). Try again.",
    "swap": "The swap stopped without finishing (Modal's time limit, a restart, or no credit left). Start it again on the "
            "same job: it resumes from the last finished clip."}


def add_silent_audio(path, probe, run=subprocess.run):
    """VHS_LoadVideo always pulls the audio track and fails on a silent clip (found in the first live run). If the
    source has no audio, mux in a silent AAC track (video stream copied, nothing re-encoded). Returns True if it changed the file."""
    if probe(path).get("has_audio"):
        return False
    path = Path(path)
    tmp = path.with_name("with_audio_" + path.name)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise JobError("That video has no sound and a silent track could not be added. Use an MP4 with audio.")
    tmp.replace(path)
    return True


def new_h3_job(root, video, photo, photo_ext, consent, probe, **settings):
    """Store the source video and the new person's photo, freeze the settings, return the job id. Nothing runs yet.
    `probe(path) -> {"duration": seconds, ...}` reads the video (swapkit.probe in production; a fake in tests)."""
    s = normalize_settings(**settings)
    job = tsvc.new_job(root, video, photo, photo_ext, consent, video_name=VIDEO_NAME,
                       meta={"kind": KIND, "engine": "minimax-h3", "prepare": {"state": "idle"}})
    d = tsvc.job_dir(root, job)
    try:
        try:
            duration = float(probe(d / VIDEO_NAME)["duration"])
        except JobError:
            raise
        except Exception:
            raise JobError("That video could not be read. Use an MP4.")
        s["start"], s["length"] = plan_section(duration, s["start"], s["length"])
        add_silent_audio(d / VIDEO_NAME, probe)
        meta = tsvc.read_meta(d)
        meta.update(request=s, duration=duration, clips=clip_count(s["length"], s["clip_seconds"]),
                    check={"state": "idle"}, swap={"state": "idle", "attempts": 0})
        tsvc.write_meta(d, meta)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return job


def _h3_meta(root, job):
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    if meta.get("kind") != KIND:
        raise JobError("This is not an H3 job.", 409)
    return d, meta


def start_check(root, job):
    """Queue the mask check (cheap, SAM only). May run again any time the swap is not running."""
    d, meta = _h3_meta(root, job)
    if meta["check"].get("state") in ACTIVE["check"]:
        raise JobError("The mask check is already running.", 409)
    if meta["swap"].get("state") in ACTIVE["swap"]:
        raise JobError("The swap is running. Wait for it to finish.", 409)
    tsvc.set_stage(d, "check", state="queued", message=None, started=time.time())


def start_swap(root, job, require_check=True):
    """Queue the paid swap. Allowed from "idle" (first run) or "error" (a RESUME: the same job keeps its comfy_out folder,
    so the Context Loop continues from the last finished clip). Refused while running or once done (no double billing).
    Returns True when this is a resume."""
    d, meta = _h3_meta(root, job)
    state = meta["swap"].get("state", "idle")
    if state in ACTIVE["swap"]:
        raise JobError("The swap is already running for this clip.", 409)
    if state == "done":
        raise JobError("This clip is already swapped. Start a new job to swap again.", 409)
    if require_check and meta["check"].get("state") != "ready":
        raise JobError("Check the mask first: the swap needs it.", 409)
    attempts = int(meta["swap"].get("attempts", 0))
    resume = state == "error" and attempts > 0
    tsvc.set_stage(d, "swap", state="queued", message=None, attempts=attempts + 1, resume=resume, started=time.time())
    return resume


def begin(root, job, stage):
    """The GPU function picked the job up."""
    d, meta = _h3_meta(root, job)
    if meta[stage].get("state") != "queued":
        raise JobError(f"The {stage} step is not queued.", 409)
    tsvc.set_stage(d, stage, state="running" if stage == "check" else "generating")


def finish(root, job, stage, produced):
    """Store the file ComfyUI made as check.mp4 / h3.mp4 in the job folder and mark the stage finished."""
    d, _ = _h3_meta(root, job)
    name = CHECK_FILE if stage == "check" else SWAP_FILE
    tmp = d / (name + ".tmp")
    shutil.copyfile(produced, tmp)
    tmp.replace(d / name)
    tsvc.set_stage(d, stage, state="ready" if stage == "check" else "done", file=name,
                   bytes=(d / name).stat().st_size, finished=time.time())


def fail(root, job, stage, message):
    """`message` must be safe to show (no paths, no internals)."""
    d, _ = _h3_meta(root, job)
    tsvc.set_stage(d, stage, state="error", message=message)


def _expire_stuck(d, meta, now):
    for stage, active in ACTIVE.items():
        st = meta.get(stage) or {}
        if st.get("state") in active and now - (st.get("at") or meta["created"]) > STUCK_AFTER_SECONDS[stage]:
            meta = tsvc.set_stage(d, stage, state="error", message=STUCK_MESSAGE[stage])
    return meta


def status(root, job, now=None):
    d, _ = _h3_meta(root, job)
    m = _expire_stuck(d, tsvc.read_meta(d), time.time() if now is None else now)
    return {"job": job, "engine": m.get("engine"), "check": m["check"], "swap": m["swap"], "request": m["request"],
            "clips": m.get("clips"), "expires_at": m["created"] + tsvc.JOB_TTL_SECONDS}


def result_path(root, job, stage):
    d, meta = _h3_meta(root, job)
    name = CHECK_FILE if stage == "check" else SWAP_FILE
    done = "ready" if stage == "check" else "done"
    if meta[stage].get("state") != done or not (d / name).exists():
        raise JobError(f"The {stage} result is not ready yet.", 409)
    return d / name


def stage_inputs(d, meta):
    """Copy the uploads into the job's ComfyUI input folder. Returns (video_name, picture_name) as the loaders see them."""
    inp = Path(d) / IN_DIR
    inp.mkdir(exist_ok=True)
    video_name, picture_name = "source.mp4", "picture" + Path(meta["ref"]).suffix
    for src, dst in ((VIDEO_NAME, video_name), (meta["ref"], picture_name)):
        if not (inp / dst).exists() or (inp / dst).stat().st_size != (Path(d) / src).stat().st_size:
            shutil.copyfile(Path(d) / src, inp / dst)
    return video_name, picture_name


def graph_for(root, job, stage):
    """The exact graph a stage submits, also saved as graph_<stage>.json in the job folder for debugging."""
    d, meta = _h3_meta(root, job)
    video_name, picture_name = stage_inputs(d, meta)
    s = meta["request"]
    g = check_graph(s, video_name) if stage == "check" else swap_graph(s, video_name, picture_name)
    (d / f"graph_{stage}.json").write_text(json.dumps(g, indent=1))
    return g


# ---------------------------------------------------------------- ComfyUI results

def history_state(entry):
    """"running" | "success" | "error" for one /history/<prompt_id> entry (None = not in history yet)."""
    if not entry:
        return "running"
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        return "error"
    if st.get("completed") or st.get("status_str") == "success":
        return "success"
    return "running"


def history_error(entry):
    """A short reason from an errored history entry (for logs only; never shown to the user as is)."""
    for msg in (entry.get("status") or {}).get("messages") or []:
        if isinstance(msg, list) and len(msg) == 2 and msg[0] == "execution_error":
            info = msg[1] or {}
            return f"{info.get('node_type', '?')} ({info.get('node_id', '?')}): {info.get('exception_message', '')}"[:500]
    return "ComfyUI reported an error."


def output_files(entry, node_id, folders):
    """Files a node reported in its history outputs ("images" / "gifs" / "videos" lists of {filename, subfolder, type}),
    resolved against `folders` = {"output": Path, "temp": Path}. Only existing files inside those folders are returned."""
    out = []
    for items in ((entry or {}).get("outputs") or {}).get(str(node_id), {}).values():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or "filename" not in it:
                continue
            base = Path(folders.get(it.get("type", "output"), folders["output"])).resolve()
            p = (base / (it.get("subfolder") or "") / it["filename"]).resolve()
            if base in p.parents and p.is_file():
                out.append(p)
    return out


def newest_mp4(folder, since=0.0):
    """Fallback when a node does not report its file: the newest .mp4 under `folder` written after `since`."""
    folder = Path(folder)
    cands = [p for p in folder.rglob("*.mp4") if p.stat().st_mtime >= since] if folder.is_dir() else []
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def models_missing(models_dir, stage):
    """Model files a stage needs that are not in the models Volume yet: ["folder/file", ...]."""
    return [f"{m['folder']}/{m['file']}" for m in MODEL_FILES
            if stage in m["for"] and not (Path(models_dir) / m["folder"] / m["file"]).is_file()]


def resolve_sources(overrides=None):
    """Download sources per model file: MODEL_SOURCES with `overrides` ({file: {"repo_id", "path_in_repo"?, "revision"?}})
    on top. Raises listing every file that still has no source: nothing is guessed."""
    src = dict(MODEL_SOURCES)
    for k, v in (overrides or {}).items():
        if k not in src:
            raise ValueError(f"Unknown model file: {k}")
        if not isinstance(v, dict) or not isinstance(v.get("repo_id"), str) or "/" not in v["repo_id"]:
            raise ValueError(f"Source for {k} needs a repo_id like owner/name.")
        src[k] = {"repo_id": v["repo_id"], "path_in_repo": v.get("path_in_repo") or k, "revision": v.get("revision")}
    missing = [k for k, v in src.items() if not v]
    if missing:
        raise ValueError("No verified download source for: " + ", ".join(missing)
                         + ". Pass them with --sources (see docs/H3.md); they are not guessed.")
    return src


FETCH_FLAG = ".h3-fetch-started"            # in the models Volume root while h3_app.fetch_models runs
FETCH_STALE_SECONDS = 4 * 3600 + 300        # fetch_models' Modal timeout plus a margin: a flag older than this is a dead fetch


def weights_state(models_dir, now=None, stale=FETCH_STALE_SECONDS):
    """"ready" | "downloading" | "missing" for the H3 model files (the web API's /health and /setup).
    ready = no file is missing for either stage (models_missing). downloading = files are missing but a fetch wrote its
    start flag less than `stale` seconds ago. Anything else (no flag, unreadable flag, a fetch that died) is missing,
    so /setup can start the fetch again; files already on the Volume are not downloaded twice."""
    if not models_missing(models_dir, "swap") and not models_missing(models_dir, "check"):
        return "ready"
    flag = Path(models_dir) / FETCH_FLAG
    if flag.is_file():
        now = time.time() if now is None else now
        try:
            started = float(flag.read_text())
        except (OSError, ValueError):
            started = flag.stat().st_mtime
        if 0 <= now - started < stale:
            return "downloading"
    return "missing"
