"""
augmentation_ablation.py — Train Model A with one augmentation removed at a time.

Tells you which of the 5 augmentations actually contribute to anomaly detection
performance and which physical symmetries the model relies on.

Five variants trained (2M, 50 epochs each, same as baseline):
  - no_rotate:    remove rotation augmentation
  - no_translate: remove translation augmentation
  - no_smear:     remove pT smearing
  - no_softdrop:  remove soft drop (constituent dropping)
  - no_collinear: remove collinear splitting

After training, run ablation_eval.py on each variant checkpoint to get AUC.
This script just handles training — evaluation is done separately.

Usage:
    # Train one variant
    python scripts/train/augmentation_ablation.py --remove rotate

    # Train all variants (run this 5 times with different --remove)
    for aug in rotate translate smear softdrop collinear; do
        screen -dmS ablation_$aug bash -c "source venv/bin/activate && \\
          python scripts/train/augmentation_ablation.py --remove $aug \\
          2>&1 | tee logs/ablation_$aug.log"
    done

    # Then evaluate each
    for aug in rotate translate smear softdrop collinear; do
        python scripts/eval/ablation_eval.py \\
            --ckpt_A data/checkpoints/ablation/no_$aug/encoder_epoch050.pt \\
            --ckpt_B2 data/checkpoints/phase2/model_B2_75ep/encoder_epoch075.pt \\
            --ckpt_C  data/checkpoints/phase2/model_C/autoencoder_epoch050.pt \\
            --out_dir data/results/aug_ablation/no_$aug
    done
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

from src.data.preprocessed_loader import PreprocessedJetDatasetCached
from src.models.encoder import JetEncoder, NTXentLoss
from src.augmentations.jet_augmentations import JetAugmentation

VALID_AUGS = ["rotate", "translate", "smear", "softdrop", "collinear"]

AUG_PARAM_NAMES = {
    "rotate":    "rotate_p",
    "translate": "translate_p",
    "smear":     "smear_pt_p",
    "softdrop":  "soft_drop_p",
    "collinear": "collinear_split_p",
}


def train_epoch(model, loader, optimizer, scaler, criterion, aug, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for xs, masks in tqdm(loader, leave=False):
        xs    = xs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
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
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    removed = args.remove
    print(f"Augmentation ablation: removing '{removed}'")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir) / f"no_{removed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = PreprocessedJetDatasetCached(args.data, max_jets=args.max_jets)
    _nw = min(args.num_workers, 4)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=_nw, pin_memory=True,
                        prefetch_factor=2 if _nw > 0 else None,
                        persistent_workers=_nw > 0)

    # --- Augmentation: set removed aug probability to 0 ---
    aug_kwargs = {v: 1.0 if v != "smear_pt_p" else 0.5 for v in
                  ["rotate_p", "translate_p", "smear_pt_p", "soft_drop_p", "collinear_split_p"]}
    # Defaults:
    aug_kwargs["soft_drop_p"]       = 0.5
    aug_kwargs["collinear_split_p"] = 0.3
    # Zero out the removed augmentation
    aug_kwargs[AUG_PARAM_NAMES[removed]] = 0.0
    aug = JetAugmentation(**aug_kwargs)

    active = {k: v for k, v in aug_kwargs.items() if v > 0}
    print(f"Active augmentations: {active}")
    print(f"Removed: {AUG_PARAM_NAMES[removed]} (set to 0)")

    # --- Model ---
    model = JetEncoder(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
    ).to(device)

    try:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    min_lr = args.lr * 0.05
    warmup_ep = 5

    def _lr_lambda(epoch):
        if epoch < warmup_ep:
            return (epoch + 1) / max(warmup_ep, 1)
        progress = (epoch - warmup_ep) / max(args.epochs - warmup_ep, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio    = min_lr / args.lr
        return ratio + (1.0 - ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    criterion = NTXentLoss(temperature=0.07)
    scaler    = GradScaler()

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scaler, criterion, aug,
                           device, grad_clip=args.grad_clip)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss})
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={loss:.4f}  lr={lr:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = save_dir / f"encoder_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": raw.state_dict(),
                        "args": vars(args), "removed_aug": removed}, ckpt_path)
            print(f"  Saved {ckpt_path}")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Checkpoints in {save_dir}")
    print(f"\nEvaluate with:")
    print(f"  python scripts/eval/ablation_eval.py \\")
    print(f"      --ckpt_A {save_dir}/encoder_epoch{args.epochs:03d}.pt \\")
    print(f"      --ckpt_B2 data/checkpoints/phase2/model_B2_75ep/encoder_epoch075.pt \\")
    print(f"      --ckpt_C  data/checkpoints/phase2/model_C/autoencoder_epoch050.pt \\")
    print(f"      --out_dir data/results/aug_ablation/no_{removed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--remove", required=True, choices=VALID_AUGS,
                   help="Which augmentation to remove: rotate|translate|smear|softdrop|collinear")
    p.add_argument("--data",     default="data/aspen/RunG_batch0_processed.h5")
    p.add_argument("--max_jets", type=int, default=None)
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--input_dim",  type=int, default=7)
    p.add_argument("--d_model",    type=int, default=128)
    p.add_argument("--nhead",      type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--save_dir",  default="data/checkpoints/ablation")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    main(args)
