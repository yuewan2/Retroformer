import torch
import pickle
import numpy as np
from rdkit import Chem

from retroformer.utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from retroformer.utils.smiles_utils import canonical_smiles
from retroformer.utils.translate_utils import batch_infer_reaction_center
from retroformer.models.model import RetroModel
from retroformer.rdchiral.template_extractor import get_fragments_for_changed_atoms


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


class RetroFormer():
    def __init__(self, checkpoint_path=None, vocab_path=None, device='cpu'):
        self.device = device
        if vocab_path is not None:
            self.load_vocab(vocab_path)
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279))

    def load_model(self, checkpoint_path):
        self.model = RetroModel(num_layers=8, d_model=256,
                                heads=8, d_ff=2048, dropout=0,
                                vocab_size_src=len(self.vocab_src_stoi),
                                vocab_size_tgt=len(self.vocab_tgt_stoi),
                                shared_vocab=True,
                                shared_encoder=False,
                                src_pad_idx=self.vocab_src_stoi['<pad>'],
                                tgt_pad_idx=self.vocab_tgt_stoi['<pad>'])
        
        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model.to(self.device)
        self.model.eval()

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            self.vocab_src_itos, self.vocab_tgt_itos = pickle.load(f)
        self.vocab_src_stoi = {v: i for i, v in enumerate(self.vocab_src_itos)}
        self.vocab_tgt_stoi = {v: i for i, v in enumerate(self.vocab_tgt_itos)}

    def parse_inputs(self, mol_smiles):
        cano_prod = clear_map_number(mol_smiles)
        smiles_graph = SmilesGraph(cano_prod)
        src_token = ['<UNK>'] + smi_tokenizer(cano_prod)
        src_token = [self.vocab_src_stoi.get(st, self.vocab_src_stoi['<unk>']) for st in src_token]
        return cano_prod, src_token, smiles_graph

    def tag_reaction_center(self, src_tokens, rc):
        for i in range(len(src_tokens)):
            if i in rc:
                if src_tokens[i][0] == '[':
                    src_tokens[i] = src_tokens[i].replace(']', ':{}]'.format(999))
                else:
                    src_tokens[i] = '[{}:{}]'.format(src_tokens[i], 999)
        smiles = ''.join(src_tokens)
        return smiles

    def infer_reaction_center(self, atom_rc_scores, bond_rc_scores, graph_packs,
                              beta, percent_aa, percent_ab, k=5, num_removal=5, max_count=25):
        _, predicts = batch_infer_reaction_center(atom_rc_scores, bond_rc_scores, graph_packs,
                                                  alpha_atom=-1, alpha_bond=-1, beta=beta,
                                                  percent_aa=percent_aa, percent_ab=percent_ab,
                                                  verbose=False, k=k, factor_func=self.factor_func,
                                                  num_removal=num_removal, max_count=max_count)
        return predicts

    def build_rc_smarts(self, cano_prod, non_reactive_mask):
        cano_prod_explicitH = Chem.MolToSmiles(Chem.MolFromSmiles(cano_prod), allHsExplicit=True)
        prod_tokens = smi_tokenizer(cano_prod_explicitH)
        for i, token in enumerate(prod_tokens):
            if not non_reactive_mask[i]:
                if token[0] == '[' and token[-1] == ']':
                    prod_tokens[i] = prod_tokens[i].replace(']', ':{}]'.format(i))
                else:
                    prod_tokens[i] = '[{}:{}]'.format(prod_tokens[i], i)

        mol = Chem.MolFromSmiles(''.join(prod_tokens))

        atom_tags = []
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom_tags.append(atom.GetProp('molAtomMapNumber'))

        prod_rc_smarts, _, _ = get_fragments_for_changed_atoms([mol], atom_tags, category='product')
        prod_rc_smarts = prod_rc_smarts[1:-1]

        patt = Chem.MolFromSmarts(prod_rc_smarts)
        for atom in patt.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

        rc_smarts = Chem.MolToSmarts(patt)
        #
        # except:
        #     atom_to_use, atom_symbols = [], []
        #     for atom in mol.GetAtoms():
        #         if atom.HasProp('molAtomMapNumber'):
        #             atom_to_use.append(atom.GetIdx())
        #             atom.ClearProp('molAtomMapNumber')
        #         atom_symbols.append(atom.GetSmarts())
        #
        #     rc_smarts = Chem.MolFragmentToSmiles(mol, atom_to_use,
        #                                          atomSymbols=atom_symbols, allHsExplicit=True,
        #                                          isomericSmiles=True, allBondsExplicit=True)

        return rc_smarts

    def compute(self, product_smiles, verbose=False, topk=5):
        cano_prod, src_token, smiles_graph = self.parse_inputs(product_smiles)
        # Prepare Inputs:
        src = torch.LongTensor(src_token).unsqueeze(1).to(self.device)
        full_adj_matrix = torch.from_numpy(smiles_graph.full_adjacency_tensor)
        bond = torch.zeros((1, len(src_token), len(src_token), 7), dtype=torch.long).to(self.device)
        bond[0, 1:full_adj_matrix.shape[0] + 1, 1:full_adj_matrix.shape[1] + 1] = full_adj_matrix
        graph_packs = (bond, [smiles_graph])

        # Encode:
        with torch.no_grad():
            prior_encoder_out, edge_feature = self.model.encoder(src, bond)
            atom_rc_scores = self.model.atom_rc_identifier[1](self.model.atom_rc_identifier[0](prior_encoder_out) / 10)
            bond_rc_scores = self.model.bond_rc_identifier[1](self.model.bond_rc_identifier[0](edge_feature) / 10)

        predicts = self.infer_reaction_center(atom_rc_scores, bond_rc_scores, graph_packs,
                                              beta=0.5, percent_aa=0.4, percent_ab=0.55, k=topk,
                                              num_removal=5, max_count=25)

        decoded_src_token = [self.vocab_src_itos[i] for i in src_token[1:]]

        rc_candidates = []
        for j, (rc, rc_score) in enumerate(predicts[0]):
            src_smiles = self.tag_reaction_center(decoded_src_token, rc)

            if verbose:
                print(rc, rc_score)
                print(src_smiles)

            non_reactive_mask = np.ones(len(decoded_src_token), dtype=bool)
            for idx in rc:
                non_reactive_mask[idx] = 0

            rc_cands = self.build_rc_smarts(cano_prod, non_reactive_mask)
            rc_candidates.append(rc_cands)

        return rc_candidates
