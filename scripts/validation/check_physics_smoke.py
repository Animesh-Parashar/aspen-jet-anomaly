"""Gate full training on finite GPU smoke and exact checkpoint reload checks."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.models.controlled_encoder import JetBackbone,ContrastiveJetModel
from src.data.controlled_loader import ControlledDataset
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
f=a.folder;manifest=json.loads((f/'COMPLETE.json').read_text());cfg=manifest['config']
assert manifest['smoke'] and manifest['device']=='cuda' and len(manifest['history'])==1
# Trusted checkpoint created by this scheduled job; optimizer/RNG need pickle.
c=torch.load(f/'checkpoint.pt',map_location='cpu',weights_only=False)
w=torch.load(f/'last.pt',map_location='cpu',weights_only=True)
assert c['epoch']==1 and c['config']==w['config']==cfg
assert c['model'].keys()==w['model'].keys()
assert all(torch.equal(v,w['model'][k]) for k,v in c['model'].items())
old=torch.load('data/results/convergence_extensions_26172.rachel/aspen_contrastive/initial_backbone.pt',map_location='cpu',weights_only=True)
new=torch.load(f/'initial_backbone.pt',map_location='cpu',weights_only=True)
assert old.keys()==new.keys() and all(torch.equal(v,new[k]) for k,v in old.items())
d=ControlledDataset(cfg['aspen_root'],'aspen','validation',16,cfg['sample_seed'])
x,m=d.x[:8],d.mask[:8]
models=[ContrastiveJetModel(JetBackbone(**cfg['backbone'])).eval() for _ in range(2)]
for model,state in zip(models,[c['model'],w['model']]):model.load_state_dict(state)
with torch.inference_mode():
    z=[model.encode(x,m) for model in models]
assert torch.isfinite(z[0]).all()
torch.testing.assert_close(z[0],z[1],rtol=0,atol=0)
(f/'SMOKE_VERIFIED.json').write_text(json.dumps({'status':'passed','initial_backbone_matches_control':True,'checkpoint_reload_exact_cpu':True,'scope':'reload equivalence; not GPU interrupted-resume equivalence'},indent=2)+'\n')
print('SMOKE_VERIFIED',f)
