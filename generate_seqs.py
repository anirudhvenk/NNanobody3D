import torch
import pandas as pd
from graph_generation.data import *
from graph_generation.hierarchical import *
from ensemble_stack.inference import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data = AntibodyDataset('graph_generation/data/seeds/3eak.jsonl', cdr_type='3')

model_ckpt, opt_ckpt, model_args = torch.load(
    'graph_generation/weights/model.best', map_location=device)
model = HierarchicalDecoder(model_args).to(device)
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(model_ckpt)
optimizer.load_state_dict(opt_ckpt)
model.eval()

hX, hS, hL, hmask = completize(data.data)

with torch.no_grad():
    cdrs = []
    enrichment_scores = []
    for i in range(100):
        cdr3, ppl, _ = model.generate(hS, hL, hmask, return_ppl=True)
        enrichment = get_stacked_prediction(cdr3).item()

        if (enrichment > 0.5):
            print(cdr3, enrichment)
            cdrs.append(cdr3[0])
            enrichment_scores.append(enrichment)

    generated = pd.DataFrame({'cdr3': cdrs, 'enrichment': enrichment_scores})
    generated.to_csv(
        'graph_generation/data/generated/3eak_generated.tsv', index=False, sep='\t')
