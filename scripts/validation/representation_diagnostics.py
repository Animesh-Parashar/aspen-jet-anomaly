"""Validation-only geometry audit; no test, signal, reference, or model selection.

Reports descriptive geometry, not anomaly-detection performance or significance.
Runs on scheduler-allocated GPU; CPU is reserved for unit tests of pure functions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.controlled_loader import ControlledDataset
from src.models.controlled_encoder import JetBackbone
from src.augmentations.controlled_augmentations import rotate


def geometry(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite matrix with at least two rows')
    centered = z - z.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / (len(z)-1)).clip(0)
    total = eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 0]
    weights = positive / total if total > 0 else positive
    norms = np.linalg.norm(z, axis=1)
    return dict(total_variance=float(total),
        participation_rank=float(total**2 / (eigenvalues @ eigenvalues)) if total > 0 else 0.,
        entropy_rank=float(np.exp(-(weights*np.log(weights)).sum())) if total > 0 else 0.,
        top_component_fraction=float(eigenvalues[-1]/total) if total > 0 else None,
        eigenvalues_descending=eigenvalues[::-1].tolist(),
        norm_quantiles=np.quantile(norms, [0, .05, .5, .95, 1]).tolist())


def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


@torch.inference_mode()
def encode(model, x, mask):
    return torch.cat([model(x[i:i+128].cuda(), mask[i:i+128].cuda()).cpu()
                      for i in range(0, len(x), 128)]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jets', type=int, default=4096)
    args = parser.parse_args()
    if args.out.exists() or args.jets < 128:
        raise ValueError('Require fresh output and at least 128 validation jets')
    if not torch.cuda.is_available():
        raise RuntimeError('Requires scheduler GPU; refusing silent CPU fallback')
    torch.set_num_threads(1)
    torch.manual_seed(20260929)
    names = ['aspen_contrastive', 'aspen_reconstruction', 'lhco_contrastive']
    manifests = {name: json.loads((args.runs/name/'COMPLETE.json').read_text()) for name in names}
    cfg = manifests[names[0]]['config']
    models, hashes = {}, {}
    initial = None
    for name in names:
        folder = args.runs/name
        manifest = manifests[name]
        if manifest['smoke'] or manifest['config'] != cfg or len(manifest['history']) != cfg['epochs']:
            raise ValueError('Require completed, matched, non-smoke runs')
        for source, expected in manifest['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Training source changed: {source}')
        checkpoint = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != cfg or checkpoint['domain'] != manifest['domain'] or checkpoint['objective'] != manifest['objective'] or checkpoint['smoke']:
            raise ValueError('Checkpoint/manifest mismatch')
        backbone = JetBackbone(**cfg['backbone'])
        backbone.load_state_dict({k.removeprefix('backbone.'): v for k,v in checkpoint['model'].items() if k.startswith('backbone.')}, strict=True)
        models[name] = backbone.eval().cuda()
        state = torch.load(folder/'initial_backbone.pt', map_location='cpu', weights_only=True)
        if initial is not None and (state.keys() != initial.keys() or any(not torch.equal(v, initial[k]) for k,v in state.items())):
            raise ValueError('Initial backbones differ')
        initial = state
        hashes[name] = hashlib.sha256((folder/'last.pt').read_bytes()).hexdigest()
    random_model = JetBackbone(**cfg['backbone'])
    random_model.load_state_dict(initial, strict=True)
    models['random_shared_initialization'] = random_model.eval().cuda()
    results = {}
    for domain in ['aspen', 'lhco']:
        data = ControlledDataset(cfg[domain+'_root'], domain, 'validation', args.jets, cfg['sample_seed'])
        x, mask = data.x, data.mask
        # Difference of redundant logarithms recovers the preprocessing pT denominator,
        # not the scalar sum of retained constituent pT or a cross-domain calibration.
        observables = dict(log_preprocessing_jet_pt=(x[:,0,3]-x[:,0,2]).numpy(),
            retained_multiplicity=(~mask).sum(1).numpy(),
            retained_pt_fraction=x[...,2].exp().masked_fill(mask,0).sum(1).numpy())
        angles = torch.linspace(-torch.pi, torch.pi, len(x))
        rotated, rotated_mask = rotate(x, mask, angles)
        # Only pairs from distinct source events are used for the distance denominator.
        order = np.random.default_rng(20260929).permutation(len(x))
        distinct = np.any(data.event_ids != data.event_ids[order], axis=1)
        if not distinct.any():
            raise ValueError('Need distinct events for between-jet distances')
        result = dict(provenance=data.provenance, unique_events=len(np.unique(data.event_ids, axis=0)),
            event_ids_sha256=hashlib.sha256(data.event_ids.tobytes()).hexdigest(),
            rotation_no_change_fraction=float((rotated == x).all(2).all(1).float().mean()),
            observable_quantiles={k: np.quantile(v,[0,.05,.5,.95,1]).tolist() for k,v in observables.items()}, models={})
        for name, model in models.items():
            z = encode(model, x, mask)
            zr = encode(model, rotated, rotated_mask)
            within = np.sum((z-zr)**2,axis=1)
            between = np.sum((z[distinct]-z[order][distinct])**2,axis=1)
            report = geometry(z)
            report['rotation_mse_over_distinct_event_mse'] = float(within.mean()/between.mean()) if between.mean()>0 else None
            report['norm_spearman'] = {k:rho(np.linalg.norm(z,axis=1), v) for k,v in observables.items()}
            result['models'][name] = report
        results[domain] = result
    output = dict(stage='validation_geometry_only', checkpoint_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, numpy=np.__version__, config=cfg, results=results,
        limitations=['Descriptive, jet-weighted diagnostics; no confidence intervals or signal performance.',
        'Nonzero effective rank does not establish useful anomaly sensitivity.',
        'Norm correlations do not measure all information carried by the embedding.',
        'Rotation distance tests backbone geometry, not just the trained projection.',
        'Full covariance spectra are saved; there is no selected rank threshold.',
        'Identically named pT features do not establish matched detector calibration.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps({domain: r['models'] for domain,r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
