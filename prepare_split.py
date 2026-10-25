import random
import shutil
from pathlib import Path

def make_local_split(src="train", dst=".", frac=0.15, seed=42):
    """Salin 15% gambar per kelas dari train ke my_val/ (sekali saja), sisanya ke my_train/."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        print(f"[INFO] Folder '{dst_va}' sudah ada. Tidak melakukan split ulang.")
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    print(f"[SUKSES] Selesai membagi data: {dst_va} (15%) dan {dst_tr} (85%) dengan seed={seed}.")
    return dst_tr, dst_va

if __name__ == "__main__":
    make_local_split()
