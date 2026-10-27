# Train a Raptor-family "student" CoAtNet on lonnieqin's public raptor-core tensors (96 x 352 px, the exact geometry our
# Raptor stage infers with) with nartaa's public trainer (train_knee.py, Apache-2.0) run unchanged except three wrapper patches:
# the four tensor shards stand in for the corpus, and the backbone is built at the training resolution.
# Labels: 0.5 x JEV-blend soft labels + 0.5 x our Platt-calibrated 4-fold OOF predictions (no OAI; gold-58 held out).
#   modal run modal/train_student.py --tag stu_c352_mixcal --labels labels_mix05cal.parquet
import modal

app = modal.App("rsna-student")
vol = modal.Volume.from_name("rsna-knee-tensors")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "timm", "pandas", "pyarrow", "scikit-learn", "numpy<2.3", "huggingface_hub", "safetensors")
         .add_local_file("outputs/research/fresh_1021/modal_probe/train_knee.py", "/root/work/train_knee.py")
         .add_local_dir("modal/data", "/root/data"))

RUNNER = r'''
import os, sys, json, time, shutil, importlib.util
import numpy as np, pandas as pd
RES = int(os.environ["STU_RES"])
t0 = time.time()
parts, masks, ids = [], [], []
for k in range(4):
    src = f"/vol/s{k}"
    dst = f"/root/tensors/vol_s{k}.u8.npy"
    if not os.path.exists(dst):
        shutil.copyfile(f"{src}/vol_s{k}.u8.npy", dst)
    v = np.load(dst, mmap_mode="r")
    m = np.load(f"{src}/mask_s{k}.u8.npy")
    ix = pd.read_csv(f"{src}/index_s{k}.csv", dtype={"StudyInstanceUID": str}).sort_values("row")
    assert (ix["row"].values == np.arange(len(ix))).all() and len(ix) == v.shape[0] == m.shape[0], (k, len(ix), v.shape, m.shape)
    assert ix["done"].astype(int).eq(1).all()
    parts.append(v); masks.append(m); ids += ix["StudyInstanceUID"].tolist()
    print(f"[tensors] shard {k}: {v.shape} {v.dtype} mask {m.shape} {m.dtype} sum {int(m.sum())}", flush=True)
print(f"[tensors] {len(ids)} studies ready in {time.time() - t0:.0f}s", flush=True)


class MultiVols:
    def __init__(self, ps):
        self.ps = ps
        self.off = np.cumsum([0] + [p.shape[0] for p in ps])
        self.shape = (int(self.off[-1]),) + tuple(ps[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        k = int(np.searchsorted(self.off, row, side="right") - 1)
        return self.ps[k][row - self.off[k]]


VOLS, MASKS, IDS = MultiVols(parts), np.concatenate(masks, axis=0), np.array(ids)
spec = importlib.util.spec_from_file_location("train_knee", "/root/work/train_knee.py")
tk = importlib.util.module_from_spec(spec)
sys.modules["train_knee"] = tk
spec.loader.exec_module(tk)
tk._open_corpus = lambda: (VOLS, MASKS)
tk._open_ids = lambda: IDS
_orig_bb = tk.build_backbone


def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    if arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_")):
        import timm
        return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3, global_pool="avg", img_size=RES)
    return _orig_bb(arch, pretrained)


tk.build_backbone = build_backbone
sys.argv = ["train_knee.py"] + sys.argv[1:]
tk.main()
'''


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=4 * 3600, volumes={"/vol": vol},
              ephemeral_disk=524288)
def train(tag: str, labels: str, res: int = 352, epochs: int = 16, seed: int = 7, extra: str = ""):
    import os, subprocess, glob, shutil, time
    os.makedirs("/root/tensors", exist_ok=True)
    shutil.copyfile("/root/data/train.csv", "/root/train.csv")      # trainer reads RSNA/train.csv with RSNA = dirname(HERE)
    open("/root/work/run_student.py", "w").write(RUNNER)
    argv = ["--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "--res", str(res), "--norm", "imagenet",
            "--workers", "12", "--k", "12", "--k_eval", "24", "--bs", "8", "--accum", "2", "--grad_ckpt",
            "--epochs", str(epochs), "--seed", str(seed), "--bb_lr", "4.01e-05", "--head_lr", "0.00195", "--wd", "0.0588",
            "--labels", f"/root/data/{labels}", "--tag", tag] + (extra.split() if extra else [])
    env = dict(os.environ, STU_RES=str(res), PYTHONUNBUFFERED="1")
    t0 = time.time()
    rc = subprocess.run(["python", "/root/work/run_student.py"] + argv, cwd="/root/work", env=env).returncode
    out = f"/vol/out/{tag}"
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(f"/root/work/raptor_*{tag}*"):
        shutil.copy(f, out)
    vol.commit()
    return {"rc": rc, "minutes": round((time.time() - t0) / 60, 1), "files": sorted(os.listdir(out))}


@app.local_entrypoint()
def main(tag: str = "stu_c352_mixcal", labels: str = "labels_mix05cal.parquet", res: int = 352, epochs: int = 16,
         seed: int = 7, extra: str = ""):
    print(train.remote(tag, labels, res, epochs, seed, extra), flush=True)
