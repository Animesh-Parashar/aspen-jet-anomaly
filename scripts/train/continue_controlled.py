"""Controlled development training: no signal/reference/test access or AUC selection."""
import argparse
import copy
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.models.controlled_encoder import JetBackbone, ContrastiveJetModel, ReconstructionJetModel
from src.models.encoder import NTXentLoss
from src.augmentations.controlled_augmentations import CoreAugmentation


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_rng(recovery, generator):
    torch.set_rng_state(recovery['torch_rng'])
    if recovery['cuda_rng']:
        torch.cuda.set_rng_state_all(recovery['cuda_rng'])
    generator.set_state(recovery['loader_rng'])
    np.random.set_state(recovery['numpy_rng'])
    random.setstate(recovery['python_rng'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--domain', choices=['aspen', 'lhco'], required=True)
    p.add_argument('--objective', choices=['contrastive', 'reconstruction', 'random'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--smoke', action='store_true', help='Two train/validation batches; never a scientific result')
    p.add_argument('--resume', type=Path, required=True, help='Trusted local completed run directory')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU requested but unavailable; refusing silent CPU training')
    if args.device == 'cpu' and not args.smoke:
        raise ValueError('Only tiny CPU smoke runs are supported')
    if args.out.exists():
        raise ValueError('Output exists; refusing overwrite')
    torch.set_num_threads(1)
    seed_all(cfg['training_seed'])
    device = torch.device(args.device)
    if args.smoke:
        cfg.update(train_jets=32, validation_jets=16, batch_size=8, epochs=1)
    train = ControlledDataset(cfg[args.domain+'_root'], args.domain, 'train', cfg['train_jets'], cfg['sample_seed'])
    valid = ControlledDataset(cfg[args.domain+'_root'], args.domain, 'validation', cfg['validation_jets'], cfg['sample_seed'])
    if set(map(tuple, train.event_ids)) & set(map(tuple, valid.event_ids)):
        raise ValueError('Training/validation event overlap')
    backbone = JetBackbone(**cfg['backbone'])
    initial_backbone = copy.deepcopy(backbone.state_dict())
    model = (ReconstructionJetModel(backbone) if args.objective == 'reconstruction'
             else ContrastiveJetModel(backbone)).to(device)
    # Reset streams after head creation so initialization changes do not change data order.
    seed_all(cfg['training_seed']+1)
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(cfg['training_seed']))
    val_loader = DataLoader(valid, batch_size=cfg['batch_size'], shuffle=False, drop_last=True, num_workers=0)
    if not len(loader) or not len(val_loader):
        raise ValueError('Need at least one full batch')
    aug = CoreAugmentation(cfg['split_probability'])
    loss_fn = NTXentLoss(cfg['temperature'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    parent = json.loads((args.resume/'COMPLETE.json').read_text())
    for source, expected in parent['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError('Parent training source changed: '+source)
    if parent['smoke'] or args.smoke or args.objective == 'random':
        raise ValueError('Resume requires a completed non-smoke trained model')
    if parent['domain'] != args.domain or parent['objective'] != args.objective:
        raise ValueError('Parent task mismatch')
    if {k:v for k,v in cfg.items() if k != 'epochs'} != {k:v for k,v in parent['config'].items() if k != 'epochs'}:
        raise ValueError('Only total epochs may change on continuation')
    # Only deserialize our explicitly verified, trusted local recovery artifact.
    recovery = torch.load(args.resume/'checkpoint.pt', map_location='cpu', weights_only=False)
    if recovery['config'] != parent['config'] or recovery['history'] != parent['history']:
        raise ValueError('Recovery checkpoint does not match completed parent')
    if recovery['epoch'] != len(parent['history']) or cfg['epochs'] <= recovery['epoch']:
        raise ValueError('Require a strictly longer completed-epoch budget')
    if recovery['domain'] != args.domain or recovery['objective'] != args.objective or recovery['smoke']:
        raise ValueError('Recovery task mismatch')
    if train.provenance != parent['train'] or valid.provenance != parent['validation']:
        raise ValueError('Dataset provenance changed')
    final = torch.load(args.resume/'last.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, final['model'][k]) for k,v in recovery['model'].items()):
        raise ValueError('Recovery and final model differ')
    model.load_state_dict(recovery['model'])
    optimizer.load_state_dict(recovery['optimizer'])
    initial_backbone = torch.load(args.resume/'initial_backbone.pt', map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True)
    sources = [Path(__file__), Path('src/models/controlled_encoder.py'), Path('src/models/encoder.py'),
               Path('src/augmentations/controlled_augmentations.py'), Path('src/data/controlled_loader.py')]
    manifest = dict(config=cfg, objective=args.objective, domain=args.domain, smoke=args.smoke,
                    train=train.provenance, validation=valid.provenance,
                    source_sha256={str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                    torch=torch.__version__, numpy=np.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                    parameters=sum(p.numel() for p in model.parameters()),
                    backbone_parameters=sum(p.numel() for p in backbone.parameters()))
    manifest['continuation'] = dict(parent=str(args.resume.resolve()), parent_epoch=recovery['epoch'], recovery_sha256=hashlib.sha256((args.resume/'checkpoint.pt').read_bytes()).hexdigest())
    (args.out/'RUNNING.json').write_text(json.dumps(manifest, indent=2)+'\n')
    torch.save(initial_backbone, args.out/'initial_backbone.pt')
    start = time.perf_counter()
    history = copy.deepcopy(recovery['history'])
    restore_rng(recovery, loader.generator)
    for epoch in range(recovery['epoch'], cfg['epochs']):
        epoch_start = time.perf_counter()
        metrics = dict(epoch=epoch+1)
        for phase, batches in [('train', loader), ('validation', val_loader)]:
            training = phase == 'train'
            model.train(training)
            total, count = 0., 0
            # Fixed validation views each epoch, with no effect on train RNG streams.
            rng = torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else [])
            with rng if not training else contextlib.nullcontext():
                if not training:
                    torch.manual_seed(88173)
                with torch.set_grad_enabled(training):
                    for step, (x, mask) in enumerate(batches):
                        if args.smoke and step >= 2:
                            break
                        x, mask = x.to(device), mask.to(device)
                        if args.objective == 'contrastive':
                            (x1,m1),(x2,m2) = aug(x,mask)
                            loss = loss_fn(model(x1,m1),model(x2,m2))
                        else:
                            prediction, _ = model(x,mask)
                            loss = model.reconstruction_scores(prediction,x,mask).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite loss')
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                            optimizer.step()
                        total += loss.item()*len(x)
                        count += len(x)
            metrics[phase+'_loss'] = total/count
        if device.type == 'cuda':
            torch.cuda.synchronize()
        metrics['epoch_seconds'] = time.perf_counter()-epoch_start
        history.append(metrics)
        # Atomically replace the last completed epoch; preserve a usable checkpoint
        # if PBS terminates a later epoch. Includes optimizer and RNG for recovery.
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            config=cfg, objective=args.objective, domain=args.domain, smoke=args.smoke,
            epoch=epoch+1, history=history, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            loader_rng=loader.generator.get_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate())
        torch.save(checkpoint, args.out/'checkpoint.tmp')
        (args.out/'checkpoint.tmp').replace(args.out/'checkpoint.pt')
        progress = dict(epoch=epoch+1, total_epochs=cfg['epochs'], history=history,
                        elapsed_seconds=time.perf_counter()-start)
        (args.out/'progress.tmp').write_text(json.dumps(progress, indent=2)+'\n')
        (args.out/'progress.tmp').replace(args.out/'progress.json')
        print(json.dumps(metrics), flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Last fixed epoch only: never select by held-out signal performance.
    torch.save(dict(model=model.state_dict(), config=cfg, objective=args.objective,
                    domain=args.domain, smoke=args.smoke), args.out/'last.pt')
    result = dict(manifest, history=history, elapsed_seconds=time.perf_counter()-start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
    (args.out/'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.out/'RUNNING.json').unlink()
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'], peak_gpu_bytes=result['peak_gpu_bytes'])))


if __name__ == '__main__':
    main()
