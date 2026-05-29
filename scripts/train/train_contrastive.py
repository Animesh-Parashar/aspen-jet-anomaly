"""
Contrastive pretraining — works for both Aspen (real) and preprocessed sim data.

Phase 2 recommended settings:
    # Model A — real data (single batch, ~2M jets)
    python train_contrastive.py \
        --data      data/aspen/RunG_batch0_processed.h5 \
        --data_type aspen \
        --epochs 50 --batch_size 4096 \
        --d_model 128 --nhead 8 --num_layers 4 --latent_dim 256 \
        --save_dir  data/checkpoints/phase2/model_A

    # Model B2 — LHCO sim baseline
    python train_contrastive.py \
        --data      data/lhco/lhco_bg_jets.h5 \
        --data_type sim \
        --epochs 50 --batch_size 4096 \
        --d_model 128 --nhead 8 --num_layers 4 --latent_dim 256 \
        --save_dir  data/checkpoints/phase2/model_B2

    # Model A scale — real data, 6 batches (~10M jets)
    python train_contrastive.py \
        --data data/aspen/RunG_batch0_processed.h5 \
               data/aspen/RunG_batch1_processed.h5 \
               data/aspen/RunG_batch2_processed.h5 \
               data/aspen/RunG_batch3_processed.h5 \
               data/aspen/RunG_batch4_processed.h5 \
               data/aspen/RunG_batch5_processed.h5 \
        --data_type aspen \
        --epochs 50 --batch_size 4096 \
        --d_model 128 --nhead 8 --num_layers 4 --latent_dim 256 \
        --save_dir  data/checkpoints/phase3/model_A_10M
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
import json
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.aspen_loader import AspenJetDataset, N_FEATURES
from src.data.preprocessed_loader import PreprocessedJetDataset, PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder, NTXentLoss
from src.augmentations.jet_augmentations import JetAugmentation


def train_epoch(model, loader, optimizer, scaler, criterion, aug, device,
                feature_cols=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for xs, masks in tqdm(loader, leave=False):
        # non_blocking overlaps H→D transfer with CPU-side prefetch of next batch
        xs    = xs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if feature_cols is not None:
            xs = xs[:, :, feature_cols]
        (x1, m1), (x2, m2) = aug(xs, masks)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            z1   = model(x1, m1)
            z2   = model(x2, m2)
            loss = criterion(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed input shapes → let cuDNN find the fastest conv/attention kernels once
    torch.backends.cudnn.benchmark = True
    # TF32 matmuls on Ada/Ampere: same exponent range as fp32 but 10× faster
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory // 1024**3} GB VRAM)")
    print(f"Device: {device}  |  data_type: {args.data_type}  |  bf16 AMP: True")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset (always use cached loader: loads into RAM, then zero I/O) ---
    from torch.utils.data import ConcatDataset

    def _load_one(data_path, max_j):
        from pathlib import Path as _P
        if args.data_type == "aspen" and not _P(data_path).stem.endswith("_processed"):
            print(f"TIP: run preprocess_aspen.py once on {data_path} for ~30x faster loading.")
            return AspenJetDataset(data_path, max_jets=max_j,
                                    max_constituents=args.max_constituents)
        return PreprocessedJetDatasetCached(data_path, max_jets=max_j)

    if len(args.data) == 1:
        dataset = _load_one(args.data[0], args.max_jets)
    else:
        datasets = [_load_one(dp, args.max_jets) for dp in args.data]
        dataset  = ConcatDataset(datasets)
        print(f"Multi-batch dataset: {len(dataset):,} jets from {len(args.data)} files")

    # num_workers > 0 for async prefetch, but keep it small — __getitem__ is a
    # zero-copy torch.from_numpy() on cached RAM data, so extra workers add IPC
    # overhead not throughput.  4 workers + prefetch_factor=2 keeps the GPU fed.
    _nw = min(args.num_workers, 4)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=_nw, pin_memory=True,
                        prefetch_factor=2 if _nw > 0 else None,
                        persistent_workers=_nw > 0)

    # --- Model ---
    input_dim = len(args.feature_cols) if args.feature_cols is not None else N_FEATURES
    model = JetEncoder(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {n_params:,}  |  batch_size: {args.batch_size}  "
          f"|  negatives per step: {2*args.batch_size - 2}")

    # torch.compile traces the model once and emits optimised Triton/CUDA kernels.
    # First epoch is slow (compilation); subsequent epochs are 15-25% faster.
    try:
        torch._dynamo.config.suppress_errors = True  # OOM during compile → eager fallback
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR schedule: linear warmup then cosine decay to min_lr (never reaches 0)
    min_lr    = args.min_lr if args.min_lr is not None else args.lr * 0.05
    warmup_ep = args.warmup_epochs

    def _lr_lambda(epoch):
        # epoch is 0-indexed here (called after each step, starting at 0)
        if epoch < warmup_ep:
            return (epoch + 1) / max(warmup_ep, 1)
        progress = (epoch - warmup_ep) / max(args.epochs - warmup_ep, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio    = min_lr / args.lr
        return ratio + (1.0 - ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    criterion = NTXentLoss(temperature=args.temperature)
    aug       = JetAugmentation()
    scaler    = GradScaler()

    # --- Resume from checkpoint ---
    start_epoch = 1
    history = []
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward scheduler to match resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from {args.resume_from}  (epoch {ckpt['epoch']}, "
              f"restarting at {start_epoch}, lr={scheduler.get_last_lr()[0]:.2e})")

    # Divergence threshold: NT-Xent max = log(2B-1); flag if loss exceeds 60% of max
    _max_loss   = torch.log(torch.tensor(2 * args.batch_size - 1, dtype=torch.float)).item()
    _div_thresh = 0.6 * _max_loss
    _div_streak = 0

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scaler, criterion, aug, device,
                           feature_cols=args.feature_cols, grad_clip=args.grad_clip)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss})
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={loss:.4f}  lr={current_lr:.2e}")

        if device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  peak VRAM: {mem_gb:.1f} GB")
            torch.cuda.reset_peak_memory_stats()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = save_dir / f"encoder_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": raw.state_dict(),
                        "args": vars(args)}, ckpt_path)
            print(f"  Saved {ckpt_path}")

        # Divergence detection: stop early if loss explodes for 3 consecutive epochs
        if loss > _div_thresh:
            _div_streak += 1
            print(f"  WARNING: loss {loss:.4f} > divergence threshold "
                  f"{_div_thresh:.4f} (streak {_div_streak}/3)")
            if _div_streak >= 3:
                print(f"  DIVERGED — stopping early at epoch {epoch}. "
                      f"Resume from last good checkpoint with a lower --lr.")
                break
        else:
            _div_streak = 0

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      nargs="+", required=True,
                   help="One or more preprocessed HDF5 files (concatenated for multi-batch)")
    p.add_argument("--data_type", default="aspen", choices=["aspen", "sim"],
                   help="aspen = AspenOpenJets HDF5; sim = preprocessed jet HDF5")
    p.add_argument("--max_jets",  type=int, default=None,
                   help="None = use all jets in file")
    p.add_argument("--max_constituents", type=int, default=50)
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--batch_size",type=int, default=8192,
                   help="8192 uses ~20 GB VRAM on RTX 6000 Ada; try 16384 if memory allows")
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--d_model",   type=int, default=128)
    p.add_argument("--nhead",     type=int, default=8)
    p.add_argument("--num_layers",type=int, default=4)
    p.add_argument("--latent_dim",type=int, default=256)
    p.add_argument("--save_dir",  default="data/checkpoints/phase2/model_A")
    p.add_argument("--save_every",type=int, default=10)
    p.add_argument("--num_workers",type=int, default=4,
                   help="Capped at 4 internally for cached datasets — IPC overhead dominates beyond that")
    p.add_argument("--resume_from", default=None,
                   help="Resume from a checkpoint .pt file (model weights + epoch; optimizer resets)")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear LR warmup epochs before cosine decay (default 5)")
    p.add_argument("--min_lr", type=float, default=None,
                   help="Minimum LR floor for cosine decay (default 5%% of --lr)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm (default 1.0)")
    p.add_argument("--feature_cols", type=int, nargs="+", default=None,
                   help="Feature column indices to use (default: all 7). "
                        "4-feature sim-available only: --feature_cols 0 1 2 3")
    args = p.parse_args()
    main(args)
