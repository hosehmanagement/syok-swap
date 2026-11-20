"""app.py needs Modal and a GPU, so it is checked as text: the fit step is wired and the Qwen pins that matter are in place."""
import ast
from pathlib import Path

SRC = (Path(__file__).parent.parent / "app.py").read_text()
TREE = ast.parse(SRC)
FUNCS = {n.name for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)}


def test_fit_functions_exist():
    assert {"fit_job", "generate_job", "_qwen_editor", "swap_job"} <= FUNCS


def test_qwen_pins_avoid_the_known_bad_combination():
    assert '"diffusers==0.37.1"' in SRC          # 0.37 is the first version that supports the 2511 transformer
    assert '"transformers==4.57.6"' in SRC       # transformers 5.x gives silently wrong edits (diffusers issue 14194)
    assert "huggingface_hub[hf_xet]<1.0" in SRC


def test_generate_is_started_through_the_coordinator():
    assert "generate_job.spawn" in SRC and "fit_job.remote(job)" in SRC


def test_fit_reference_replaces_wans_reference_before_generation():
    assert 'w / "proc" / "src_ref.png"' in SRC


def test_qwen_gets_the_scene_shape_not_the_photos():
    assert "width=W, height=H" in SRC
