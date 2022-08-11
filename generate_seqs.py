import torch
import pandas as pd
from graph_generation.data import *
from graph_generation.hierarchical import *
from ensemble_stack.inference import *
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Generator:
    def __init__(self, model_weights_path):
        data = AntibodyDataset('graph_generation/data/seeds/3eak.jsonl', cdr_type='3')
        self.hX, self.hS, self.hL, self.hmask = completize(data.data)

        model_ckpt, opt_ckpt, model_args = torch.load(model_weights_path, map_location=device)
        self.model = HierarchicalDecoder(model_args).to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.load_state_dict(model_ckpt)
        optimizer.load_state_dict(opt_ckpt)
        self.model.eval()
        
    

    def generate_sequences(self, num_sequences, threshold,  generation):
        with torch.no_grad():
            cdrs = []
            enrichment_scores = []
            for i in tqdm(range(num_sequences)):
                cdr3, ppl, _ = self.model.generate(self.hS, self.hL, self.hmask, return_ppl=True)
                enrichment = get_stacked_prediction(cdr3).item()

                if (enrichment > threshold):
                    cdrs.append(cdr3[0])
                    enrichment_scores.append(enrichment)

            generated = pd.DataFrame({'cdr3': cdrs, 'enrichment': enrichment_scores})
            generated.to_csv(
                f'graph_generation/data/generated/3eak_gen_{generation}.tsv', index=False, sep='\t')
