"""
Phase 2 — Model C: Transformer Autoencoder on AspenOpenJets (real data).

Usage:
    python train_autoencoder.py \
        --data data/aspen/RunG_batch0_processed.h5 \
        --epochs 50 --batch_size 2048 \
        --save_dir data/checkpoints/phase2/model_C
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.aspen_loader import N_FEATURES
from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.autoencoder import JetAutoencoder, masked_mse_loss


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for xs, masks in tqdm(loader, leave=False):
        xs    = xs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x_hat, _ = model(xs, masks)
            loss = masked_mse_loss(x_hat, xs, masks)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory // 1024**3} GB VRAM)")
    print(f"Using device: {device}  |  bf16 AMP: True")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = PreprocessedJetDatasetCached(args.data, max_jets=args.max_jets)
    _nw = min(args.num_workers, 4)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=_nw, pin_memory=True,
                         prefetch_factor=2 if _nw > 0 else None,
                         persistent_workers=_nw > 0)

    model = JetAutoencoder(
        input_dim=N_FEATURES,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_constituents=args.max_constituents,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model C params: {n_params:,}  |  batch_size: {args.batch_size}")

    try:
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss})
        print(f"Epoch {epoch:03d}/{args.epochs}  recon_loss={loss:.6f}")

        if device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  peak VRAM: {mem_gb:.1f} GB")
            torch.cuda.reset_peak_memory_stats()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt = save_dir / f"autoencoder_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": raw.state_dict(),
                        "args": vars(args)}, ckpt)
            print(f"  Saved {ckpt}")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True)
    p.add_argument("--max_jets", type=int, default=None)
    p.add_argument("--max_constituents", type=int, default=50)
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--d_model",  type=int, default=64)
    p.add_argument("--nhead",    type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--save_dir", default="data/checkpoints/phase2/model_C")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    main(args)
