"""H3 engine (h3_engine.py): openjutsu's Simple swap + Check mask graphs, patched in Python, and the job folder / meta.json
rules around them. No GPU, no network, no Modal: the workflows are the bundled exports and the GPU steps are fakes.

The graph tests mirror openjutsu test/simple.test.ts case by case, so the port keeps the same behaviour."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_engine as h3
import texture_service as svc
from texture_service import JobError
from test_swapkit import make_video
from test_texture_service import SKIN, photo_bytes

GRAPH = h3.load_workflow("simple")
MASK = h3.load_workflow("mask")
HAT_TYPES = ["CLIPTextEncode", "SAM3_VideoTrack", "SAM3_TrackToMask", "GrowMaskWithBlur", "MaskComposite", "MaskComposite"]


def settings(hat_fix, no_lip_sync=True, **kw):
    base = dict(role=" the  singer ", resolution="480p", motion=512, hat_fix=hat_fix, no_lip_sync=no_lip_sync,
                margin="loose", target="cat", seed=7, clip_seconds=6, start=0.0, length=0.0)
    base.update(kw)
    return base


def by_class(g, t):
    return [n for n in g.values() if n["class_type"] == t]


def is_link_to(g, value, t):
    return isinstance(value, list) and g.get(value[0], {}).get("class_type") == t


# ---------------------------------------------------------------- the bundled exports

def test_bundled_exports_are_openjutsus_files():
    """The copies are byte-identical to openjutsu's exports at commit 7fe118b (checked when they were copied)."""
    import hashlib
    sums = {"simple": "24dc5b8ef45788c839beb96b617b404308ddfd8921e89cfd9008442ad29a666a",
            "mask": "86d7902c94a557eaf03fa1440bca53724ccc3939d2a3b7927890e0216102e4d7"}
    for name, want in sums.items():
        assert hashlib.sha256((h3.WORKFLOW_DIR / f"{name}.api.json").read_bytes()).hexdigest() == want
    assert (h3.WORKFLOW_DIR / "LICENSE").read_text().startswith("MIT License")


def test_load_workflow_refuses_other_names():
    with pytest.raises(ValueError):
        h3.load_workflow("../app")


def test_node_ids_match_the_export():
    types = {"picture": "LoadImage", "video": "VHS_LoadVideo", "plan": "H3OneClickSwapPlan",
             "loopStart": "MiniMaxH3ChainLoopStart", "peopleMask": "H3PeopleMask", "growMask": "GrowMaskWithBlur",
             "loopEnd": "MiniMaxH3ChainLoopEnd", "final": "MiniMaxH3ChainAssemble"}
    for key, t in types.items():
        assert GRAPH[h3.NODES[key]]["class_type"] == t, key
    assert [GRAPH[i]["class_type"] for i in h3.NODES["hat"]] == HAT_TYPES
    assert sorted(MASK[i]["class_type"] for i in h3.MASK_NODES["hat"]) == sorted(HAT_TYPES)
    assert MASK[h3.MASK_NODES["video"]]["class_type"] == "VHS_LoadVideo"
    assert MASK[h3.MASK_NODES["preview"]]["class_type"] == "VHS_VideoCombine"
    for g, i in ((GRAPH, h3.NODES["target"]), (MASK, h3.MASK_NODES["target"])):
        assert (g[i]["class_type"], g[i]["inputs"]["text"]) == ("CLIPTextEncode", "person")


def test_model_manifest_matches_the_loaders_in_the_exports():
    """Every file a loader names in the two exports is in MODEL_FILES, in the right folder, and nothing else is."""
    loader_folder = {"UNETLoader": ("unet_name", "diffusion_models"), "LoraLoaderModelOnly": ("lora_name", "loras"),
                     "CLIPLoader": ("clip_name", "text_encoders"), "VAELoader": ("vae_name", "vae"),
                     "CheckpointLoaderSimple": ("ckpt_name", "checkpoints")}
    named = set()
    for g in (GRAPH, MASK):
        for n in g.values():
            if n["class_type"] in loader_folder:
                key, folder = loader_folder[n["class_type"]]
                named.add((folder, n["inputs"][key]))
    assert named == {(m["folder"], m["file"]) for m in h3.MODEL_FILES}
    check_needs = {(n["class_type"]) for n in MASK.values() if n["class_type"] in loader_folder}
    assert check_needs == {"CheckpointLoaderSimple"}
    assert [m["file"] for m in h3.MODEL_FILES if "check" in m["for"]] == ["sam3.1_multiplex_fp16.safetensors"]


# ---------------------------------------------------------------- prune_simple (simple.test.ts)

def test_without_hat_fix_the_hat_nodes_go_and_the_grow_mask_reads_the_people_mask():
    pruned = h3.prune_simple(GRAPH, settings(False))
    for i in h3.NODES["hat"]:
        assert i not in pruned
    assert pruned[h3.NODES["growMask"]]["inputs"]["mask"] == [h3.NODES["peopleMask"], 0]
    plan = pruned[h3.NODES["plan"]]["inputs"]
    assert [plan["resolution"], plan["seed"], plan["clip_seconds"], plan["roles"]] == ["480p", 7, 6, "the singer"]
    assert pruned[h3.NODES["growMask"]]["inputs"]["expand"] == 60
    assert pruned[h3.NODES["target"]]["inputs"]["text"] == "cat"
    assert h3.NODES["hat"][0] in GRAPH, "the source graph is not modified"


def test_a_custom_size_crops_the_source_frames_before_anything_reads_them():
    def readers(g, nid):
        return sum(1 for n in g.values() for v in n["inputs"].values() if h3.is_link(v) and v[0] == nid and v[1] == 0)
    pruned = h3.prune_simple(GRAPH, settings(True, resolution={"width": 1248, "height": 736}))
    assert pruned["custom_size"]["inputs"] == {"image": [h3.NODES["video"], 0], "upscale_method": "bilinear",
                                               "width": 1248, "height": 736, "crop": "center"}
    assert readers(pruned, h3.NODES["video"]) == 1
    assert readers(pruned, "custom_size") == readers(GRAPH, h3.NODES["video"])
    assert pruned[h3.NODES["plan"]]["inputs"]["short_edge"] == 736
    assert "custom_size" not in h3.prune_simple(GRAPH, settings(True))
    assert h3.canvas_size(640, 360, "736p") == [1312, 736]
    assert h3.canvas_size(1080, 1920, "480p") == [480, 864]


def test_a_part_of_the_video_becomes_vhs_skip_and_frame_cap():
    pruned = h3.prune_simple(GRAPH, settings(False))
    h3.set_section(pruned, h3.NODES["video"], 3.5, 4)
    v = pruned[h3.NODES["video"]]["inputs"]
    assert [v["skip_first_frames"], v["frame_load_cap"]] == [84, 96]
    h3.set_section(pruned, h3.NODES["video"], 0, 0)
    assert [v["skip_first_frames"], v["frame_load_cap"]] == [0, 0]
    h3.set_section(pruned, h3.NODES["video"], 0.0625, 2.0625)      # 1.5 and 49.5 frames: JS Math.round rounds half up
    assert [v["skip_first_frames"], v["frame_load_cap"]] == [2, 50]


def test_a_bigger_motion_reference_gets_its_own_run_name():
    pruned = h3.prune_simple(GRAPH, settings(False, motion=768))
    sizes = [n["inputs"]["largest_size"] for n in by_class(pruned, "ImageScaleToMaxDimension")
             if "motion reference" in n["_meta"]["title"].lower()]
    assert sizes == [768]
    run = by_class(pruned, "MiniMaxH3ChainPlanModern")[0]["inputs"]["run_name"]
    assert pruned[run[0]]["inputs"] == {"string_a": [h3.NODES["plan"], 3], "string_b": "_motion768", "delimiter": ""}
    plain = h3.prune_simple(GRAPH, settings(False))
    assert by_class(plain, "MiniMaxH3ChainPlanModern")[0]["inputs"]["run_name"] == [h3.NODES["plan"], 3]


def test_with_hat_fix_the_full_graph_is_kept():
    pruned = h3.prune_simple(GRAPH, settings(True))
    assert len(pruned) == len(GRAPH)
    assert pruned[h3.NODES["growMask"]]["inputs"]["mask"] == [h3.NODES["hat"][5], 0]


def test_without_no_lip_sync_h3_hears_the_soundtrack_and_the_mannequin_gets_no_faces():
    pruned = h3.prune_simple(GRAPH, settings(True, no_lip_sync=False))
    assert len(pruned) == len(GRAPH) - 6
    assert by_class(pruned, "AudioAdjustVolume") == []
    assert not any("faces" in n.get("_meta", {}).get("title", "") for n in pruned.values())
    ref, = by_class(pruned, "MiniMaxH3ReferenceToVideo")
    src, = by_class(pruned, "H3SourceSlice")
    mannequin, = by_class(pruned, "H3GreyMannequins")
    assert pruned[ref["inputs"]["ref_videos.ref_video_0"][0]]["class_type"] == "H3GreyMannequins"
    assert pruned[ref["inputs"]["ref_video_audios.ref_video_audio_0"][0]]["class_type"] == "MiniMaxH3ChainCurrent"
    assert pruned[src["inputs"]["source_audio"][0]]["class_type"] == "TrimAudioDuration"
    assert "faces" not in mannequin["inputs"] and "face_size" not in mannequin["inputs"]
    assert "no_lip_sync" not in pruned[h3.NODES["plan"]]["inputs"]


def test_with_no_lip_sync_h3_hears_silence_and_the_plan_keeps_the_flag():
    pruned = h3.prune_simple(GRAPH, settings(False, no_lip_sync=True))
    ref, = by_class(pruned, "MiniMaxH3ReferenceToVideo")
    src, = by_class(pruned, "H3SourceSlice")
    for link in (ref["inputs"]["ref_video_audios.ref_video_audio_0"], src["inputs"]["source_audio"]):
        assert pruned[link[0]]["inputs"]["volume"] == -100
    flag = pruned[pruned[h3.NODES["plan"]]["inputs"]["no_lip_sync"][0]]
    assert (flag["class_type"], flag["inputs"]["value"]) == ("PrimitiveBoolean", True)
    assert is_link_to(pruned, by_class(pruned, "H3GreyMannequins")[0]["inputs"]["faces"], "H3PeopleMask")


def test_every_combination_leaves_no_dangling_link():
    for hat in (True, False):
        for lip in (True, False):
            for motion in h3.MOTION_SIZES:
                for res in ("480p", {"width": 512, "height": 768}):
                    g = h3.prune_simple(GRAPH, settings(hat, lip, motion=motion, resolution=res))
                    h3.check_links(g)
    with pytest.raises(ValueError, match="removed node"):
        g = json.loads(json.dumps(GRAPH))
        del g["2"]
        h3.check_links(g)


def test_drop_no_lip_sync_refuses_an_unknown_link_to_a_removed_node():
    g = json.loads(json.dumps(GRAPH))
    g["x"] = {"class_type": "PreviewAny", "inputs": {"source": ["68", 0]}}
    with pytest.raises(ValueError, match="x.source"):
        h3.drop_no_lip_sync(g)


def test_motion_size_and_run_name_need_their_nodes():
    with pytest.raises(ValueError, match="motion reference"):
        h3.set_motion_size({"1": {"class_type": "LoadImage", "inputs": {}}}, 768)
    with pytest.raises(ValueError, match="Production Plan"):
        h3.suffix_run_name({}, "x", "_a", "t")


# ---------------------------------------------------------------- prune_mask

def test_the_mask_check_drops_the_hat_nodes_without_hat_fix():
    pruned = h3.prune_mask(MASK, False, "tight", "dog")
    assert pruned[h3.MASK_NODES["growMask"]]["inputs"]["expand"] == 20
    assert pruned[h3.MASK_NODES["target"]]["inputs"]["text"] == "dog"
    for i in h3.MASK_NODES["hat"]:
        assert i not in pruned
    assert pruned[h3.MASK_NODES["growMask"]]["inputs"]["mask"] == [h3.MASK_NODES["peopleMask"], 0]
    assert len(h3.prune_mask(MASK, True, "normal", "person")) == len(MASK)
    assert h3.MASK_NODES["hat"][0] in MASK


# ---------------------------------------------------------------- full graphs with the files bound

def test_swap_graph_binds_the_files_and_the_section():
    s = h3.normalize_settings(margin="tight", start=1.5, length=4, seed=99, resolution="736p")
    g = h3.swap_graph(s, "source.mp4", "picture.png")
    v, p = g[h3.NODES["video"]]["inputs"], g[h3.NODES["picture"]]["inputs"]
    assert (v["video"], v["skip_first_frames"], v["frame_load_cap"], p["image"]) == ("source.mp4", 36, 96, "picture.png")
    plan = g[h3.NODES["plan"]]["inputs"]
    assert (plan["seed"], plan["resolution"], plan["roles"]) == (99, "736p", "")
    assert g[h3.NODES["growMask"]]["inputs"]["expand"] == 20
    assert GRAPH[h3.NODES["video"]]["inputs"]["video"] == ""


def test_check_graph_uses_the_same_section_and_mask_settings():
    s = h3.normalize_settings(margin="loose", start=2, length=3, hat_fix=True, target="black cat")
    g = h3.check_graph(s, "source.mp4")
    v = g[h3.MASK_NODES["video"]]["inputs"]
    assert (v["video"], v["skip_first_frames"], v["frame_load_cap"]) == ("source.mp4", 48, 72)
    assert g[h3.MASK_NODES["growMask"]]["inputs"]["expand"] == 60
    assert g[h3.MASK_NODES["target"]]["inputs"]["text"] == "black cat"
    assert len(g) == len(MASK)


def test_bind_inputs_checks_the_node_types():
    g = json.loads(json.dumps(GRAPH))
    with pytest.raises(ValueError):
        h3.bind_inputs(g, h3.NODES["picture"], "a.mp4")
    with pytest.raises(ValueError):
        h3.bind_inputs(g, h3.NODES["video"], "a.mp4", h3.NODES["video"], "b.png")


# ---------------------------------------------------------------- clip plan (clips.ts)

def test_unknown_duration_and_invalid_clip_lengths_do_not_enter_the_loop():
    for d in (math.inf, math.nan, -1, 0):
        assert h3.clip_count(d, 6) == 0
        assert h3.clip_starts(d, 6) == [0]
    for length in (math.inf, math.nan, 0, 39 / 24):
        assert h3.clip_count(12, length) == 0


def test_clip_count_follows_the_swap_plan_grid():
    assert h3.clip_count(6, 6) == 1
    assert h3.clip_count(16, 6) == 4
    assert h3.clip_count(30, 6) == 7
    assert h3.clip_count(10, 15) == 1


def test_clip_starts_follow_the_39_frame_overlap():
    assert h3.clip_starts(339 / 24, 6) == [0, 102, 204]
    assert h3.clip_count(107 / 24, 3) == 3
    assert h3.clip_starts(107 / 24, 3) == [0, 17, 68]
    assert h3.clip_starts(16, 6) == [0, 102, 204, 306]


# ---------------------------------------------------------------- settings

def test_settings_defaults_and_normalisation():
    s = h3.normalize_settings()
    assert s == {"margin": "normal", "start": 0.0, "length": 0.0, "seed": 0, "resolution": "480p", "role": "",
                 "target": "person", "hat_fix": False, "no_lip_sync": False, "motion": 512, "clip_seconds": 6.0}
    assert h3.normalize_settings(margin=" Loose ", target="  ", seed="12")["margin"] == "loose"
    assert h3.normalize_settings(target="  black   cat ")["target"] == "black cat"
    assert h3.normalize_settings(seed=h3.MAX_SEED)["seed"] == h3.MAX_SEED


@pytest.mark.parametrize("kw", [dict(margin="huge"), dict(seed=-1), dict(seed=h3.MAX_SEED + 1), dict(seed=1.5),
                                dict(seed="x"), dict(motion=640), dict(clip_seconds=2), dict(clip_seconds=16),
                                dict(start=-1), dict(length="nan"), dict(resolution="720p"), dict(resolution="100x900"),
                                dict(resolution="5000x720"), dict(target="x" * 81), dict(role="r" * 201)])
def test_bad_settings_are_refused(kw):
    with pytest.raises(JobError):
        h3.normalize_settings(**kw)


def test_custom_resolution_is_rounded_to_32():
    assert h3.parse_resolution("1250x740") == {"width": 1248, "height": 736}
    assert h3.parse_resolution({"width": 256, "height": 2048}) == {"width": 256, "height": 2048}
    assert h3.parse_resolution("") == "480p"
    for p in h3.PRESETS:
        assert h3.parse_resolution(p) == p


def test_plan_section():
    assert h3.plan_section(10, 0, 0) == (0.0, 10.0)
    assert h3.plan_section(10, 3, 100) == (3.0, 7.0)
    assert h3.plan_section(40, 5, 30) == (5.0, 30.0)
    for d, s, l in ((10, 9, 0), (10, 0, 1.5), (60, 0, 0), (0, 0, 0)):
        with pytest.raises(JobError):
            h3.plan_section(d, s, l)


# ---------------------------------------------------------------- job folder / meta.json

def fake_probe(seconds):
    return lambda path: {"duration": seconds}


@pytest.fixture
def clip_bytes(tmp_path):
    return make_video(tmp_path / "orig.mp4", SKIN, n=12, fps=24, w=64, h=64).read_bytes()


def new_job(tmp_path, clip_bytes, seconds=8.0, **kw):
    root = tmp_path / "jobs"
    root.mkdir(exist_ok=True)
    return root, h3.new_h3_job(root, clip_bytes, photo_bytes(), ".png", "yes", fake_probe(seconds), **kw)


def test_new_job_freezes_the_settings_in_meta(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes, margin="tight", start=1, length=4, seed=5, resolution="1080p")
    d = svc.job_dir(root, job)
    meta = svc.read_meta(d)
    assert meta["kind"] == "h3" and meta["engine"] == "minimax-h3"
    assert meta["request"]["start"] == 1.0 and meta["request"]["length"] == 4.0
    assert (meta["request"]["seed"], meta["request"]["resolution"], meta["request"]["margin"]) == (5, "1080p", "tight")
    assert meta["check"] == {"state": "idle"} and meta["swap"] == {"state": "idle", "attempts": 0}
    assert meta["prepare"]["state"] == "idle"                      # the texture slider is not started by the H3 engine
    assert meta["clips"] == h3.clip_count(4.0, 6.0) == 1
    assert (d / "raw.mp4").exists() and (d / meta["ref"]).exists()


def test_new_job_refuses_bad_input_and_stores_nothing(tmp_path, clip_bytes):
    root = tmp_path / "jobs"
    root.mkdir()
    with pytest.raises(JobError, match="consent|agreed"):
        h3.new_h3_job(root, clip_bytes, photo_bytes(), ".png", "", fake_probe(8))
    with pytest.raises(JobError, match="Margin"):
        h3.new_h3_job(root, clip_bytes, photo_bytes(), ".png", "yes", fake_probe(8), margin="x")
    with pytest.raises(JobError, match="limit"):
        h3.new_h3_job(root, clip_bytes, photo_bytes(), ".png", "yes", fake_probe(90))
    with pytest.raises(JobError, match="could not be read"):
        h3.new_h3_job(root, clip_bytes, photo_bytes(), ".png", "yes", lambda p: 1 / 0)
    assert list(root.iterdir()) == []


def test_check_then_swap_then_done(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes)
    with pytest.raises(JobError, match="Check the mask"):
        h3.start_swap(root, job)
    h3.start_check(root, job)
    with pytest.raises(JobError, match="already running"):
        h3.start_check(root, job)
    h3.begin(root, job, "check")
    assert h3.status(root, job)["check"]["state"] == "running"
    out = make_video(tmp_path / "c.mp4", SKIN, n=4, fps=24, w=32, h=32)
    h3.finish(root, job, "check", out)
    assert h3.status(root, job)["check"]["state"] == "ready"
    assert h3.result_path(root, job, "check").read_bytes() == out.read_bytes()
    with pytest.raises(JobError, match="not ready"):
        h3.result_path(root, job, "swap")
    assert h3.start_swap(root, job) is False
    with pytest.raises(JobError, match="already running"):
        h3.start_swap(root, job)
    with pytest.raises(JobError, match="swap is running"):
        h3.start_check(root, job)
    h3.begin(root, job, "swap")
    assert h3.status(root, job)["swap"]["state"] == "generating"
    h3.finish(root, job, "swap", out)
    st = h3.status(root, job)
    assert st["swap"]["state"] == "done" and st["swap"]["file"] == "h3.mp4" and st["swap"]["attempts"] == 1
    with pytest.raises(JobError, match="already swapped"):
        h3.start_swap(root, job)


def test_a_failed_swap_resumes_on_the_same_job(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes)
    h3.start_check(root, job); h3.begin(root, job, "check")
    h3.finish(root, job, "check", make_video(tmp_path / "c.mp4", SKIN, n=4, fps=24, w=32, h=32))
    h3.start_swap(root, job); h3.begin(root, job, "swap")
    d = svc.job_dir(root, job)
    (d / "comfy_out").mkdir()
    (d / "comfy_out" / "clip_001.ckpt").write_text("kept")      # what the Context Loop leaves after a finished clip
    h3.fail(root, job, "swap", "The swap failed.")
    assert h3.status(root, job)["swap"]["message"] == "The swap failed."
    assert h3.start_swap(root, job) is True                       # a resume
    st = h3.status(root, job)["swap"]
    assert (st["state"], st["attempts"], st["resume"], st["message"]) == ("queued", 2, True, None)
    assert (d / "comfy_out" / "clip_001.ckpt").read_text() == "kept"


def test_begin_needs_a_queued_stage(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes)
    with pytest.raises(JobError, match="not queued"):
        h3.begin(root, job, "check")


def test_a_step_that_died_silently_is_reported(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes)
    h3.start_check(root, job); h3.begin(root, job, "check")
    h3.finish(root, job, "check", make_video(tmp_path / "c.mp4", SKIN, n=4, fps=24, w=32, h=32))
    h3.start_swap(root, job); h3.begin(root, job, "swap")
    at = svc.read_meta(svc.job_dir(root, job))["swap"]["at"]
    assert h3.status(root, job, now=at + 60)["swap"]["state"] == "generating"
    st = h3.status(root, job, now=at + h3.STUCK_AFTER_SECONDS["swap"] + 1)["swap"]
    assert st["state"] == "error" and "resumes" in st["message"]
    assert h3.start_swap(root, job) is True


def test_wan_jobs_are_not_h3_jobs(tmp_path, clip_bytes):
    root = tmp_path / "jobs"
    root.mkdir()
    job = svc.new_job(root, clip_bytes, photo_bytes(), ".png", "yes")
    with pytest.raises(JobError, match="not an H3 job"):
        h3.status(root, job)


def test_graph_for_stages_the_inputs_and_saves_the_graph(tmp_path, clip_bytes):
    root, job = new_job(tmp_path, clip_bytes, start=1, length=3, seed=3)
    g = h3.graph_for(root, job, "swap")
    d = svc.job_dir(root, job)
    assert (d / "comfy_in" / "source.mp4").read_bytes() == (d / "raw.mp4").read_bytes()
    assert (d / "comfy_in" / "picture.png").exists()
    assert g[h3.NODES["video"]]["inputs"]["video"] == "source.mp4"
    assert g[h3.NODES["picture"]]["inputs"]["image"] == "picture.png"
    assert json.loads((d / "graph_swap.json").read_text()) == g
    c = h3.graph_for(root, job, "check")
    assert c[h3.MASK_NODES["video"]]["inputs"]["skip_first_frames"] == 24
    assert (d / "graph_check.json").exists()


# ---------------------------------------------------------------- ComfyUI history and files

def test_history_state_and_error():
    assert h3.history_state(None) == "running"
    assert h3.history_state({"status": {"status_str": "success", "completed": True}}) == "success"
    assert h3.history_state({"status": {"completed": False}}) == "running"
    err = {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {}], ["execution_error", {"node_id": "60", "node_type": "SamplerCustomAdvanced",
                                                      "exception_message": "CUDA out of memory"}]]}}
    assert h3.history_state(err) == "error"
    assert h3.history_error(err) == "SamplerCustomAdvanced (60): CUDA out of memory"
    assert h3.history_error({"status": {}}) == "ComfyUI reported an error."


def test_output_files_resolve_inside_the_folders_only(tmp_path):
    out, temp = tmp_path / "out", tmp_path / "temp"
    (out / "h3_oneclick_swap").mkdir(parents=True); temp.mkdir()
    (out / "h3_oneclick_swap" / "swap.mp4").write_bytes(b"x")
    (temp / "mask_00001.mp4").write_bytes(b"y")
    (tmp_path / "secret.mp4").write_bytes(b"z")
    entry = {"outputs": {"67": {"videos": [{"filename": "swap.mp4", "subfolder": "h3_oneclick_swap", "type": "output"}]},
                         "18": {"gifs": [{"filename": "mask_00001.mp4", "subfolder": "", "type": "temp"},
                                         {"filename": "../secret.mp4", "subfolder": "", "type": "temp"},
                                         {"filename": "missing.mp4", "subfolder": "", "type": "temp"}],
                                "text": "not a list of files"}}}
    folders = {"output": out, "temp": temp}
    assert h3.output_files(entry, "67", folders) == [(out / "h3_oneclick_swap" / "swap.mp4").resolve()]
    assert h3.output_files(entry, 18, folders) == [(temp / "mask_00001.mp4").resolve()]
    assert h3.output_files(entry, "99", folders) == []


def test_newest_mp4_fallback(tmp_path):
    import os
    assert h3.newest_mp4(tmp_path / "none") is None
    a, b = tmp_path / "a.mp4", tmp_path / "sub" / "b.mp4"
    b.parent.mkdir(); a.write_bytes(b"a"); b.write_bytes(b"b")
    os.utime(a, (1000, 1000)); os.utime(b, (2000, 2000))
    assert h3.newest_mp4(tmp_path) == b
    assert h3.newest_mp4(tmp_path, since=3000) is None


# ---------------------------------------------------------------- model files

def test_models_missing(tmp_path):
    assert h3.models_missing(tmp_path, "check") == ["checkpoints/sam3.1_multiplex_fp16.safetensors"]
    assert len(h3.models_missing(tmp_path, "swap")) == 7
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "sam3.1_multiplex_fp16.safetensors").write_bytes(b"0")
    assert h3.models_missing(tmp_path, "check") == []


def test_model_sources_are_the_verified_ones_and_never_guessed():
    # one source per model file, no more and no fewer, each a real owner/name repo with a path (verified 21 Nov 2026)
    assert set(h3.MODEL_SOURCES) == {m["file"] for m in h3.MODEL_FILES}
    for f, v in h3.MODEL_SOURCES.items():
        assert v["repo_id"].count("/") == 1 and v["path_in_repo"].endswith(f)
    assert h3.MODEL_SOURCES["sam3.1_multiplex_fp16.safetensors"]["repo_id"] == "Comfy-Org/sam3.1"
    assert h3.MODEL_SOURCES["qwen3vl_32b_minimax_h3_int4_convrot.safetensors"]["repo_id"] == "Merserk/MiniMax-H3-INT4-ConvRot"
    assert h3.MODEL_SOURCES["minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"]["repo_id"] == "Kijai/MiniMax-H3-experimental"
    assert h3.resolve_sources() == h3.MODEL_SOURCES
    with pytest.raises(ValueError, match="Unknown model file"):
        h3.resolve_sources({"other.safetensors": {"repo_id": "a/b"}})
    with pytest.raises(ValueError, match="repo_id"):
        h3.resolve_sources({"sam3.1_multiplex_fp16.safetensors": {"repo_id": "nope"}})
    src = h3.resolve_sources({"sam3.1_multiplex_fp16.safetensors": {"repo_id": "owner/repo"}})
    assert src["sam3.1_multiplex_fp16.safetensors"] == {"repo_id": "owner/repo",
                                                        "path_in_repo": "sam3.1_multiplex_fp16.safetensors", "revision": None}
    broken = dict(h3.MODEL_SOURCES, **{"sam3.1_multiplex_fp16.safetensors": None})
    old = h3.MODEL_SOURCES
    try:
        h3.MODEL_SOURCES = broken
        with pytest.raises(ValueError, match="No verified download source"):
            h3.resolve_sources()
    finally:
        h3.MODEL_SOURCES = old
