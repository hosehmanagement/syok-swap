import sys
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import consent as cs


def test_consent_real_needs_everything(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 90, np.uint8))
    with pytest.raises(PermissionError):
        cs.make_record(img, "real", attest={}, who="A", contact="a@b.c")
    full = {k: True for k in cs.REQUIRED_FOR_REAL}
    with pytest.raises(PermissionError):
        cs.make_record(img, "real", attest=full, who="", contact="")
    rec = cs.make_record(img, "real", attest=full, who="Ann", contact="ann@example.com")
    assert cs.verify(rec, img)


def test_consent_bound_to_image_and_expiry_and_revoke(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 90, np.uint8))
    other = tmp_path / "g.png"; cv2.imwrite(str(other), np.full((16, 16, 3), 91, np.uint8))
    full = {k: True for k in cs.REQUIRED_FOR_REAL}
    rec = cs.make_record(img, "real", attest=full, who="Ann", contact="x@y.z", valid_days=1, now=1000)
    with pytest.raises(PermissionError):
        cs.verify(rec, other, now=1001)
    with pytest.raises(PermissionError):
        cs.verify(rec, img, now=1000 + 2 * 86400)
    p = tmp_path / "c.json"; cs.save(rec, p); cs.revoke(p)
    with pytest.raises(PermissionError):
        cs.verify(cs.load(p), img, now=1001)


def test_consent_fictional_needs_no_person(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 5, np.uint8))
    assert cs.verify(cs.make_record(img, "fictional"), img)
