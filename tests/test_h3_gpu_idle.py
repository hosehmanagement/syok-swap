"""Every GPU function of the H3 app must switch off 60 s after its job ends (pinned, not left to Modal's default),
and the screen/controls API (web) must never hold a GPU."""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "h3_app.py"
TREE = ast.parse(SRC.read_text())


def _gpu_functions():
    out = {}
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "function":
                    out[node.name] = {k.arg: ast.unparse(k.value) for k in d.keywords}
    return out


def test_idle_default_is_60_seconds():
    src = SRC.read_text()
    assert 'GPU_IDLE_SECONDS = int(os.environ.get("SYOK_H3_GPU_IDLE_SECONDS", "60"))' in src


def test_each_gpu_job_pins_the_idle_shutdown():
    fns = _gpu_functions()
    for name in ("check_job", "swap_job", "faces_job"):
        assert name in fns, name
        assert "gpu" in fns[name], f"{name} should be a GPU function"
        assert fns[name].get("scaledown_window") == "GPU_IDLE_SECONDS", f"{name} must pin scaledown_window"


def test_no_other_function_holds_a_gpu_and_web_is_cpu():
    fns = _gpu_functions()
    gpu = sorted(n for n, kw in fns.items() if "gpu" in kw)
    assert gpu == ["check_job", "faces_job", "swap_job"], gpu
    assert "gpu" not in fns["web"] and fns["web"].get("cpu") == "2"
