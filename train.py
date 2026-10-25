import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm


def macro_f1(y, pr):
    f = []
    for c in range(5):
        tp = np.sum((pr == c) & (y == c))
        fp = np.sum((pr == c) & (y != c))
        fn = np.sum((pr != c) & (y == c))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f)), f


@torch.no_grad()
def evaluate_model(model, dl_va, dev):
    model.eval()
    ys, ps = [], []
    for x, y in dl_va:
        ps.append(model(x.to(dev)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y, pr = np.concatenate(ys), np.concatenate(ps)
    f, per = macro_f1(y, pr)
    return f, per, float((y == pr).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train/")
    p.add_argument("--val", default="my_val/")
    p.add_argument("--arch", default="convnext_tiny")
    p.add_argument("--out", default="sub_new")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="lr head; backbone = lr * bb_mult")
    p.add_argument("--bb_mult", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--aug", default="strong", choices=["basic", "strong"])
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="hanya untuk uji cepat")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    S = a.img_size

    tf_val = transforms.Compose([
        transforms.Resize(int(S * 1.14)),
        transforms.CenterCrop(S),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    aug = [
        transforms.RandomResizedCrop(S, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    ]
    if a.aug == "strong":
        aug += [transforms.RandomRotation(15), transforms.TrivialAugmentWide()]
    aug += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if a.aug == "strong":
        aug += [transforms.RandomErasing(p=0.25)]
    tf_train = transforms.Compose(aug)

    # ---- data ----
    ds_tr = datasets.ImageFolder(a.data, tf_train)
    assert ds_tr.classes == ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"], ds_tr.classes
    ds_va = datasets.ImageFolder(a.val, tf_val)
    val_keys = {(ds_va.classes[c], os.path.basename(pth)) for pth, c in ds_va.samples}
    if a.full:
        keep = list(range(len(ds_tr)))
    else:
        keep = [i for i, (pth, c) in enumerate(ds_tr.samples) if (ds_tr.classes[c], os.path.basename(pth)) not in val_keys]
    print(f"train total={len(ds_tr)} | dipakai latihan={len(keep)} | dikeluarkan (ada di my_val)={len(ds_tr)-len(keep)} | my_val={len(ds_va)}")
    if not a.full and len(keep) == len(ds_tr):
        print("PERINGATAN: tidak ada file my_val yang cocok di train/ (nama berbeda, atau my_val memang sudah dipisah dari train/). "
              "Pastikan my_val tidak ikut terlatih, kalau tidak skor my_val menjadi terlalu tinggi.")

    num_workers = 2 if os.name != "nt" else 0
    dl_tr = DataLoader(Subset(ds_tr, keep), a.bs, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=dev == "cuda")
    dl_va = DataLoader(ds_va, 64, shuffle=False, num_workers=num_workers)

    # ---- model ----
    model = timm.create_model(a.arch, pretrained=not a.no_pretrained, num_classes=5).to(dev)
    head = model.get_classifier()
    hid = {id(q) for q in head.parameters()}
    bb = [q for q in model.parameters() if id(q) not in hid]
    opt = torch.optim.AdamW([
        {"params": list(head.parameters()), "lr": a.lr},
        {"params": bb, "lr": a.lr * a.bb_mult},
    ], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=a.smooth)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{a.arch} | params={n_params/1e6:.1f}M | img_size={S} | device={dev}")

    os.makedirs(a.out, exist_ok=True)
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        sched.step()
        f, per, acc = evaluate_model(model, dl_va, dev)
        log.append({"epoch": ep + 1, "loss": tl / len(dl_tr), "val_macro_f1": f, "val_acc": acc})
        print(f"ep {ep+1:02d}/{a.epochs} loss={tl/len(dl_tr):.4f} my_val_macroF1={f:.4f} acc={acc:.4f} ({time.time()-t0:.0f}s)")
        if a.full or f > best:
            best = f
            torch.save(model.state_dict(), os.path.join(a.out, "model.pth"))

    # ---- tulis kontrak submission ----
    MODEL_PY = f'''"""Submission Studio 2 - {a.arch} (timm). get_model() hanya membangun arsitektur."""
import timm
from torchvision import transforms

ARCH = "{a.arch}"
IMG_SIZE = {S}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Urutan kelas: 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam


def get_model():
    return timm.create_model(ARCH, pretrained=False, num_classes=5)  # logits [B, 5]


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
'''
    open(os.path.join(a.out, "model.py"), "w").write(MODEL_PY)
    open(os.path.join(a.out, "requirements.txt"), "w").write("timm\n")
    json.dump({
        "args": vars(a),
        "params_M": n_params / 1e6,
        "best_my_val_macro_f1": best,
        "epochs": log,
    }, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(f"\nSelesai. Hasil di '{a.out}/'. Uji: python evaluate.py --submission ./{a.out} --data ./{a.val.rstrip('/')}")


if __name__ == "__main__":
    main()
