"""Inventory metadata and development-only physics observables, never test data.

A mass-window occupancy is not a signal yield or a resonance significance.
Producer column names: OzAmram/AOJProcessing H5_maker.py fill_jet.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with h5py.File(args.development,'r') as dev:
        if dev.attrs['partition'] != 'validation':
            raise ValueError('Only validation partition permitted')
        indices=dev['source_index'][:]
        ids=dev['event_id'][:]
    order=np.argsort(indices); indices=indices[order];ids=ids[order]
    with h5py.File(args.raw,'r') as f:
        inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in f.items() if isinstance(v,h5py.Dataset)}
        # Small development sample only, excluded from the test partition.
        kin=f['jet_kinematics'][indices]
        tagging=f['jet_tagging'][indices]
        if not np.array_equal(ids,f['event_info'][indices]):
            raise ValueError('Source event identity mismatch')
    m=kin[:,3]
    result=dict(inventory=inventory, development_rows=len(indices),development_events=len(np.unique(ids,axis=0)),
        source_indices=indices.tolist(),
        stored_jet_columns=['pt','eta','phi','CMS_softdrop_mass'],
        tagging_columns=['nConstituents','tau1','tau2','tau3','tau4','particleNet_H4qvsQCD',
         'particleNet_HbbvsQCD','particleNet_HccvsQCD','particleNet_QCD','particleNet_TvsQCD',
         'particleNet_WvsQCD','particleNet_ZvsQCD','particleNet_mass'],
        nonfinite_jet_kinematics=int((~np.isfinite(kin)).sum()),
        nonfinite_tagging=int((~np.isfinite(tagging)).sum()),
        development_mass_windows={f'{lo}_{hi}_GeV':int(((m>=lo)&(m<hi)).sum()) for lo,hi in [(60,110),(140,210)]},
        missing_analysis_inputs=['per-event trigger decisions/prescales','trigger efficiency control sample',
         'integrated luminosity certification','jet/mass calibration uncertainty variations',
         'pileup observable per event','lepton selection information'],
        limitations=['Mass-window counts include background; not identified W/Z/top yields.',
         'Development subset is not an unbiased luminosity sample; no full-data yield extrapolation.',
         'This file contains selected jets only; complete dijet events across batch boundaries are not certified.',
         'ParticleNet outputs are model predictions, not truth labels.',
         'Derived unweighted constituent mass is not CMS soft-drop mass.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_indices'},indent=2))

if __name__=='__main__':main()
