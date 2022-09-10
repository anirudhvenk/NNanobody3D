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
        
    
    def generate_sequences(self, num_sequences, threshold, generation):
        with torch.no_grad():
            all_cdrs = []
            all_ppl_scores = []
            best_cdrs = []
            best_ppl = []

            for i in tqdm(range(num_sequences)):
                cdr3, ppl, _ = self.model.generate(self.hS, self.hL, self.hmask, return_ppl=True)
                if ('#' not in cdr3[0]) and ('C' not in cdr3[0]) and ('N' not in cdr3[0]):
                    # enrichment = get_stacked_prediction(cdr3).item()

                    # if (enrichment > threshold and 'N' not in cdr3[0] and 'C' not in cdr3[0]):
                    all_cdrs.append(cdr3[0])
                    # enrichment_scores.append(enrichment)
                    all_ppl_scores.append(ppl.item())

            all_enrichment_scores = get_stacked_prediction(all_cdrs).detach().cpu().flatten()
            best_enrichment = []
            
            for cdr, enrichment, ppl in zip(all_cdrs, all_enrichment_scores, all_ppl_scores):
                if enrichment.item() > threshold:
                    best_cdrs.append(cdr)
                    best_ppl.append(ppl)
                    best_enrichment.append(enrichment.item())
            
            generated = pd.DataFrame({'cdr3': best_cdrs, 'enrichment': best_enrichment, 'perplexity': best_ppl})
            generated.to_csv(
                f'graph_generation/data/generated/3eak_gen_{generation}.tsv', index=False, sep='\t')
