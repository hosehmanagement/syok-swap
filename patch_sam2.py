"""Build-time patch: make sam2.utils.misc.get_connected_components fall back to the CPU version.
Run once inside the image after sam2 is installed:  python /opt/patch_sam2.py"""
import pathlib
import shutil
import sam2.utils.misc as misc

src = pathlib.Path(misc.__file__)
text = src.read_text()
old = "    from sam2 import _C\n\n    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())"
assert old in text, "sam2 misc.py layout changed - update patch"
new = (
    "    try:\n"
    "        from sam2 import _C\n"
    "        return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())\n"
    "    except ImportError:\n"
    "        from sam2.sam2_cc_fallback import get_connected_components_cpu\n"
    "        return get_connected_components_cpu(mask)"
)
src.write_text(text.replace(old, new))
shutil.copy(pathlib.Path(__file__).with_name("sam2_cc_fallback.py"), src.parent.parent / "sam2_cc_fallback.py")
print("patched", src)
