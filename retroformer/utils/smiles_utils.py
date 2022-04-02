import re
import rdkit
from rdkit import Chem
from retroformer.rdchiral.template_extractor import extract_from_reaction, get_changed_atoms, mols_from_smiles_list, \
    replace_deuterated

import pickle
import torch
import numpy as np
from tqdm import tqdm

BONDTYPES = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
BONDTOI = {bond: i for i, bond in enumerate(BONDTYPES)}


class SmilesGraph:
    def __init__(self, smi, existing=None):
        self.V = len(smi_tokenizer(smi))
        self.smi = smi
        if existing is None:
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = self.extract_graph_structure(smi)
        else:
            assert len(existing) == 3
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = existing

        self.full_adjacency_tensor = np.zeros((len(self.adjacency_matrix), len(self.adjacency_matrix), 7), dtype=int)
        for i in range(len(self.adjacency_matrix)):
            for j in self.adjacency_matrix[i]:
                self.full_adjacency_tensor[i][j] = self.bond_attributes[(i, j)]

    def one_hot_vector(self, val, lst):
        """Converts a value to a one-hot vector based on options in list"""
        if val not in lst:
            val = lst[-1]
        return map(lambda x: x == val, lst)

    def get_bond_attributes(self, bond):
        """
        From Neural FP defaults:
        The bond features were a concatenation of whether the bond type was single, double, triple,
        or aromatic, whether the bond was conjugated, and whether the bond was part of a ring.
        """
        # Initialize
        attributes = []
        # Add bond type
        attributes += self.one_hot_vector(
            bond.GetBondTypeAsDouble(),
            [1.0, 1.5, 2.0, 3.0]
        )
        # Add if is aromatic
        attributes.append(bond.GetIsAromatic())
        # Add if bond is conjugated
        attributes.append(bond.GetIsConjugated())
        # Add if bond is part of ring
        attributes.append(bond.IsInRing())

        return np.array(attributes)

    def extract_graph_structure(self, smi, verbose=False):
        """Build SMILES graph from molecule graph"""
        adjacency_matrix = [[] for _ in range(len(smi_tokenizer(smi)))]
        bond_types = {}
        bond_attributes = {}
        sample_mol = Chem.MolFromSmiles(smi)
        atom_ordering = [atom.GetIdx() for atom in sample_mol.GetAtoms()]
        atom_symbols = [atom.GetSmarts() for atom in sample_mol.GetAtoms()]
        neighbor_smiles_list = []
        neighbor_bonds_list, neighbor_bonds_attributes_list = [], []
        for atom in sample_mol.GetAtoms():
            neighbor_bonds = []
            neighbor_bonds_attributes = []
            atom_symbols_i = atom_symbols[:]
            atom_symbols_i[atom.GetIdx()] = '[{}:1]'.format(atom.GetSymbol())
            for i, neighbor_atom in enumerate(atom.GetNeighbors()):
                atom_symbols_i[neighbor_atom.GetIdx()] = '[{}:{}]'.format(neighbor_atom.GetSymbol(), 900 + i)
                bond = sample_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                neighbor_bonds.append(str(bond.GetBondType()))
                neighbor_bonds_attributes.append(self.get_bond_attributes(bond))

            neighbor_tagged_smiles = \
                Chem.MolFragmentToSmiles(sample_mol, atomsToUse=atom_ordering,
                                         canonical=False, atomSymbols=atom_symbols_i)
            neighbor_smiles_list.append(neighbor_tagged_smiles)
            neighbor_bonds_list.append(neighbor_bonds)
            neighbor_bonds_attributes_list.append(neighbor_bonds_attributes)

        for ni, neighbor_tagged_smiles in enumerate(neighbor_smiles_list):
            neighbor_tagged_tokens = smi_tokenizer(neighbor_tagged_smiles)
            neighbor_bonds = neighbor_bonds_list[ni]
            neighbor_bonds_attributes = neighbor_bonds_attributes_list[ni]
            cur_i, cand_js, order = -1, [], []
            for j in range(len(neighbor_tagged_tokens)):
                if re.match('\[.*:1\]', neighbor_tagged_tokens[j]):
                    cur_i = j
                if re.match('\[.*:90[0-9]\]', neighbor_tagged_tokens[j]):
                    cand_js.append(j)
                    order.append(int(re.match('\[.*:(90[0-9])\]', neighbor_tagged_tokens[j]).group(1)) - 900)
            if cur_i > -1:
                assert len(neighbor_bonds) == len(cand_js)
                neighbor_bonds = list(np.array(neighbor_bonds)[order])
                neighbor_bonds_attributes = list(np.array(neighbor_bonds_attributes)[order])
                if verbose:
                    print(neighbor_tagged_smiles)
                    print(cur_i, cand_js, neighbor_bonds, '\n')
                adjacency_matrix[cur_i] = cand_js
                for cur_j in cand_js:
                    bond_types[(cur_i, cur_j)] = BONDTOI[neighbor_bonds.pop(0)]
                    bond_attributes[(cur_i, cur_j)] = neighbor_bonds_attributes.pop(0)
        return adjacency_matrix, bond_types, bond_attributes


def set_distance(a, b):
    return np.min([len(set(a) - set(b)), len(set(b) - set(a))])


def set_overlap(a, b):
    return len(set(a).intersection(set(b)))


def select_diverse_candidate(cc_trace_with_score, diverse_k=10):
    """Select top-k diverse candidate greedily from a list of tuple of (list of nodes, scores)"""
    selected_indices = {0}
    selected_cc_trace_with_score = [cc_trace_with_score[0]]
    pair2distance = {}

    for _ in range(min(diverse_k, len(cc_trace_with_score)) - 1):
        distance = []
        explore_new = False
        for i in range(len(cc_trace_with_score)):
            if i in selected_indices:
                distance.append(-float('inf'))
                continue
            current_cc_trace, current_cc_score = cc_trace_with_score[i]
            # avg_distance, tmp_trace = [], None
            distance_min, trace_min = float('inf'), ()
            for selected_cc_trace, _ in selected_cc_trace_with_score:
                pair_key = tuple(sorted([selected_cc_trace, current_cc_trace]))
                if pair_key not in pair2distance:
                    pair2distance[pair_key] = set_distance(selected_cc_trace, current_cc_trace)

                if pair2distance[pair_key] < distance_min:
                    distance_min = pair2distance[pair_key]
                    trace_min = selected_cc_trace

            if (distance_min > 2 and set_overlap(trace_min, current_cc_trace) < 3) \
                    or ((distance_min == 2) and len(trace_min) == 2):
                distance.append(distance_min + current_cc_score)
                explore_new = True
            else:
                distance.append(-float('inf'))

        if not explore_new:
            break

        top_index = np.argsort(distance)[-1]
        selected_indices.add(top_index)
        selected_cc_trace_with_score.append(cc_trace_with_score[top_index])

    return selected_cc_trace_with_score


def get_reaction_centers_from_template(src_smiles, blank_src_smiles, graph_pack, reaction_centers):
    """Retrieve all the potential reaction center from a pool of existing molecule fragment"""
    mol_blank = Chem.MolFromSmiles(blank_src_smiles)
    mol = Chem.MolFromSmiles(src_smiles)
    if mol is None:
        return []
    potential_rcs = {}
    for rc in (reaction_centers):
        if rc[0] == '(' and rc[-1] == ')':
            rc = rc[1:-1]
        patt = Chem.MolFromSmarts(rc)
        if patt is not None and mol_blank.HasSubstructMatch(patt):
            for match in mol_blank.GetSubstructMatches(patt):
                token_match_indices = []
                for index in match:
                    atom_smarts = mol.GetAtomWithIdx(index).GetSmarts()
                    token_match_indices.append(int(re.match('.*:([0-9]+)\]', atom_smarts).group(1)))
                if tuple(sorted(token_match_indices)) not in potential_rcs:
                    score = get_cc_score(token_match_indices, graph_pack) / get_norm(token_match_indices, graph_pack)
                    potential_rcs[tuple(sorted(token_match_indices))] = score

    return sorted(potential_rcs.items(), key=lambda x: -x[1])


def dfs_cc(trace, i, visited, graph_pack, alpha_atom=0.01, alpha_bond=0.01):
    """Retrieve the connected components considering both the atom and bond reactive probability"""
    node_scores, edge_scores, adjacency_matrix, _ = graph_pack
    visited[i] = True
    neighbors = [j for j in adjacency_matrix[i] if not visited[j] and
                 node_scores[j] > alpha_atom and (edge_scores[i, j] > alpha_bond or edge_scores[j, i] > alpha_bond)]
    if not neighbors:
        return trace
    for j in neighbors:
        if not visited[j]:
            trace = dfs_cc(trace + [j], j, visited, graph_pack, alpha_atom, alpha_bond)
    return trace


def dfs_cc_atom(trace, i, visited, graph_pack, alpha_atom=0.01):
    """Retrieve the connected components considering only the atom reactive probability"""
    node_scores, edge_scores, adjacency_matrix, _ = graph_pack
    visited[i] = True
    neighbors = [j for j in adjacency_matrix[i] if not visited[j] and node_scores[j] > alpha_atom]
    if not neighbors:
        return trace
    for j in neighbors:
        if not visited[j]:
            trace = dfs_cc_atom(trace + [j], j, visited, graph_pack, alpha_atom)
    return trace


def dfs_cc_bond(trace, i, visited, graph_pack, cc_trace_parent, alpha_bond=0.01):
    """Retrieve the connected components considering only the bond reactive probability"""
    node_scores, edge_scores, adjacency_matrix, _ = graph_pack
    visited[i] = True
    neighbors = [j for j in adjacency_matrix[i] if not visited[j] and
                 (edge_scores[i, j] > alpha_bond and edge_scores[j, i] > alpha_bond) and
                 j in cc_trace_parent]

    if not neighbors:
        return trace
    for j in neighbors:
        if not visited[j]:
            trace = dfs_cc_bond(trace + [j], j, visited, graph_pack, cc_trace_parent, alpha_bond)
    return trace


def get_cc_score(cc_trace, graph_pack):
    """Retrieve the total reactive scores given a subgraph"""
    node_scores, edge_scores, adjacency_matrix, full_adjacency_matrix = graph_pack
    sub_edge_scores = edge_scores[list(cc_trace)][:, list(cc_trace)]
    sub_adj_matrix = full_adjacency_matrix[list(cc_trace)][:, list(cc_trace)]
    return sum(np.log(node_scores[list(cc_trace)])) + sum(np.log(sub_edge_scores[sub_adj_matrix > 0]))


def get_norm(cc_trace, graph_pack):
    """Retrieve the normalization factor for the normalized reactive scores"""
    node_scores, edge_scores, adjacency_matrix, full_adjacency_matrix = graph_pack
    num_nodes = len(cc_trace)
    num_edges = np.sum(full_adjacency_matrix[list(cc_trace)][:, list(cc_trace)] > 0)

    return num_nodes + num_edges


def get_boarder_cycles(cc_trace, full_adjacency_matrix):
    """Identify which nodes are within the border cycle"""
    def dfs_fc(i, trace, adj_matrix):
        neighbors = np.argwhere(adj_matrix[i] > 0).flatten()
        for j in neighbors:
            if j in trace and len(trace) - trace.index(j) > 2:
                # find cycle
                cycle = sorted(trace[trace.index(j):])
                cycles.add(tuple(cycle))
                return
            elif j not in trace:
                dfs_fc(j, trace + [j], adj_matrix)

    cycles, boarder_cycles, boarder_cycles_flatten = set(), [], []
    sub_adj_matrix = full_adjacency_matrix[cc_trace]
    sub_adj_matrix = sub_adj_matrix[:, cc_trace]
    dfs_fc(0, [0], sub_adj_matrix)

    is_cycle_boarder = [False] * len(cc_trace)
    for cycle in cycles:
        num_edges = (sub_adj_matrix[list(cycle)] > 0).sum(-1)
        for i, c in enumerate(cycle):
            if num_edges[i] < 3:
                is_cycle_boarder[cycle[i]] = True

    return is_cycle_boarder


def recursive_trim(cc_trace, cc_score_total, graph_pack, reaction_centers, total_num=None,
                   min_count=2, max_count=25, num_removal=3, depth=0):
    """Recursively prune a root graph into its sub-graphs based on heuristics"""
    if tuple(sorted(cc_trace)) in reaction_centers:
        return
    if len(cc_trace) < min_count:
        return

    node_scores, edge_scores, adj_matrix, full_adj_matrix = graph_pack
    sub_adj_matrix = full_adj_matrix[cc_trace]
    sub_adj_matrix = sub_adj_matrix[:, cc_trace]
    if total_num is None:
        total_num = (sub_adj_matrix > 0).sum().item() + len(cc_trace)

    is_boarder = ((sub_adj_matrix > 0).sum(0) == 1)
    is_cycle_boarder = get_boarder_cycles(cc_trace, full_adj_matrix)

    idx2score = {}
    cc_node_scores = node_scores[cc_trace]
    for idx in range(len(cc_trace)):
        if is_boarder[idx] or is_cycle_boarder[idx]:
            idx2score[idx] = cc_node_scores[idx]

    top_index_by_sorted_node_scores = sorted(list(idx2score.keys()), key=lambda x: idx2score[x])

    if len(cc_trace) > max_count:
        top_index_by_sorted_node_scores = top_index_by_sorted_node_scores[:1]
        depth_update = depth
    elif depth > 2:
        top_index_by_sorted_node_scores = top_index_by_sorted_node_scores[:num_removal]
        depth_update = depth + 1
    else:
        depth_update = depth + 1

    # Normalize score for reaction center
    normalize_factor = len(cc_trace) + np.sum(sub_adj_matrix > 0)
    reaction_centers[tuple(sorted(cc_trace))] = cc_score_total / normalize_factor

    for idx in top_index_by_sorted_node_scores:
        cc_trace_next = cc_trace[:]
        cc_trace_next.pop(idx)
        cc_score_next = cc_score_total
        js = [cc_trace[idx_j] for idx_j in np.argwhere(sub_adj_matrix[idx] > 0)[0]]
        cc_score_next -= np.sum(np.log(edge_scores[js, cc_trace[idx]]))
        cc_score_next -= np.sum(np.log(edge_scores[cc_trace[idx], js]))
        cc_score_next -= np.log(node_scores[cc_trace[idx]])

        recursive_trim(cc_trace_next, cc_score_next, graph_pack, reaction_centers,
                       total_num=total_num, min_count=min_count, max_count=max_count,
                       num_removal=num_removal, depth=depth_update)
    return


def get_subgraphs_by_trim(cc_trace, cc_score_total, graph_pack,
                          min_count=2, max_count=25, num_removal=3, verbose=False):
    """Wrapper for recursive pruning"""
    if num_removal < 0:
        num_removal = len(cc_trace)
    reaction_centers = {}

    recursive_trim(cc_trace, cc_score_total, graph_pack, reaction_centers,
                   min_count=min_count, max_count=max_count, num_removal=num_removal)
    if verbose:
        print('trim:')
        tmp = sorted(reaction_centers.items(), key=lambda x: -x[1])
        for t in tmp:
            print('  {}'.format(t))

    return list(reaction_centers.keys())


def add_mapping(token, map_num=1):
    """Add a given tag (atom mapping) into a SMILES token"""
    if not re.match('.*[a-zA-Z].*', token):
        return token
    if token[0] == '[':
        if re.match('\[.*:[1-9]+\]', token):
            result = token
        else:
            result = token.replace(']', ':{}]'.format(map_num))
    else:
        result = '[{}:{}]'.format(token, map_num)
    return result


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return canonical_smiles(Chem.MolToSmiles(mol))


def canonical_smiles(smi):
    """Canonicalize a SMILES without atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi


def randomize_smiles_with_am(smi):
    """Randomize a SMILES with atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    random_root = np.random.choice([(atom.GetIdx()) for atom in mol.GetAtoms()])
    return Chem.MolToSmiles(mol, rootedAtAtom=int(random_root))


def canonical_smiles_with_am(smi):
    """Canonicalize a SMILES with atom mapping"""
    atomIdx2am, pivot2atomIdx = {}, {}
    mol = Chem.MolFromSmiles(smi)
    atom_ordering = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atomIdx2am[atom.GetIdx()] = atom.GetProp('molAtomMapNumber')
            atom.ClearProp('molAtomMapNumber')
        else:
            atomIdx2am[atom.GetIdx()] = '0'
        atom_ordering.append(atom.GetIdx())

    unmapped_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ordering, canonical=False)
    mol = Chem.MolFromSmiles(unmapped_smi)
    cano_atom_ordering = list(Chem.CanonicalRankAtoms(mol))

    for i, j in enumerate(cano_atom_ordering):
        pivot2atomIdx[j + 1] = i
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)

    new_tokens = []
    for token in smi_tokenizer(Chem.MolToSmiles(mol)):
        if re.match('.*:([0-9]+)]', token):
            pivot = re.match('.*(:[0-9]+])', token).group(1)
            token = token.replace(pivot, ':{}]'.format(atomIdx2am[pivot2atomIdx[int(pivot[1:-1])]]))
        new_tokens.append(token)

    canonical_smi = ''.join(new_tokens)
    # canonical reactants order
    if '.' in canonical_smi:
        canonical_smi_list = canonical_smi.split('.')
        canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
        canonical_smi = '.'.join(canonical_smi_list)
    return canonical_smi


def smi_tokenizer(smi):
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens


def remove_am_without_canonical(smi_am, force_canonical=False):
    """Get the canonical SMILES by token modification (smiles arranged by CanonicalRankAtoms)
    :param smi_am: SMILES from `canonical_smiles_with_am`
    :param force_canonical: force the output to be canonical, not recommended since it may break the alignment
    :return:
    """

    def check_special_token(token):
        pattern = "(Mg|Zn|Si|Sn|Se|se|Ge|K|Ti|Pd|Mo|Ce|Ta|As|te|Pb|Ru|Ag|W|Pt|Co|Ca|Xe|11CH3|Rh|Tl|V|131I|Re|13c|siH|La|pH|Y|Zr|Bi|125I|Sb|Te|Ni|Fe|Mn|Cr|Al|Na|Li|Cu|nH[0-9]?|NH[1-9]?\+|\+|-|@|PH[1-9]?)"
        regex = re.compile(pattern)
        return regex.findall(token)

    new_tokens = []
    for token in smi_tokenizer(smi_am):
        # Has atommapping:
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            # print(token)
            token = token.replace(re.match('.*(:[0-9]+)]', token).group(1), '')
            explicitHs = re.match('.*(H[1-9]?).*', token)
            onlyH = re.match('\[[1-9]?H', token)
            if explicitHs and not check_special_token(token) and not onlyH:
                token = token.replace(explicitHs.group(1), '')[1:-1]
            elif not check_special_token(token) and not onlyH:
                token = token[1:-1]
            else:
                token = token

            # print(token)
            # print( )
        new_tokens.append(token)

    canonical_smi = ''.join(new_tokens)
    if force_canonical:
        canonical_smi = canonical_smiles(canonical_smi)
    return canonical_smi


def extract_relative_mapping(cano_prod_am, cano_reacts_am):
    """Extract the reactants relative positional mapping based on SMILES from `canonical_smiles_with_am`
    :param cano_prod_am:
    :param cano_reacts_am:
    :return:
    """
    cano_prod_tokens = smi_tokenizer(cano_prod_am)
    cano_reacts_tokens = smi_tokenizer(cano_reacts_am)

    # Get the exact token mapping for canonical smiles
    prodToken2posIdx = {}
    for i, token in enumerate(cano_prod_tokens):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = int(re.match('.*:([0-9]+)]', token).group(1))
            prodToken2posIdx[am] = i
        else:
            prodToken2posIdx[token] = prodToken2posIdx.get(token, []) + [i]
    position_mapping_list = []
    for i, token in enumerate(cano_reacts_tokens):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = int(re.match('.*:([0-9]+)]', token).group(1))
            prod_posIdx = prodToken2posIdx.get(am, -1)

            if prod_posIdx != -1:
                if (i, prod_posIdx) not in position_mapping_list:
                    position_mapping_list.append((i, prod_posIdx))
                    # print(i, prod_posIdx, cano_reacts_tokens[i], cano_prod_tokens[prod_posIdx])

            # Try to expand with increment
            react_pivot = i + 1
            prod_pivot = prod_posIdx + 1
            while (react_pivot, prod_pivot) not in position_mapping_list and \
                    react_pivot < len(cano_reacts_tokens) and prod_pivot < len(cano_prod_tokens) and \
                    cano_reacts_tokens[react_pivot] == cano_prod_tokens[prod_pivot]:
                # print(react_pivot, prod_pivot, cano_reacts_tokens[react_pivot])
                position_mapping_list.append((react_pivot, prod_pivot))
                react_pivot += 1
                prod_pivot += 1


            # Try to expand with decrement
            react_pivot = i - 1
            prod_pivot = prod_posIdx - 1
            while (react_pivot, prod_pivot) not in position_mapping_list and \
                    react_pivot > -1 and prod_pivot > -1 and \
                    cano_reacts_tokens[react_pivot] == cano_prod_tokens[prod_pivot]:
                position_mapping_list.append((react_pivot, prod_pivot))
                # print(react_pivot, prod_pivot, cano_reacts_tokens[react_pivot])
                react_pivot -= 1
                prod_pivot -= 1


    return position_mapping_list


def get_nonreactive_mask(cano_prod_am, raw_prod, raw_reacts, radius=0):
    """Retrieve the ground truth reaction center by RDChiral"""
    reactants = mols_from_smiles_list(replace_deuterated(raw_reacts).split('.'))
    products = mols_from_smiles_list(replace_deuterated(raw_prod).split('.'))
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    # print(changed_atom_tags)

    for _ in range(radius):
        mol = Chem.MolFromSmiles(cano_prod_am)
        changed_atom_tags_neighbor = []
        for atom in mol.GetAtoms():
            if atom.GetSmarts().split(':')[1][:-1] in changed_atom_tags:
                for n_atom in atom.GetNeighbors():
                    changed_atom_tags_neighbor.append(n_atom.GetSmarts().split(':')[1][:-1])
        changed_atom_tags = list(set(changed_atom_tags + changed_atom_tags_neighbor))
    # print(changed_atom_tags)

    nonreactive_mask = []
    for i, token in enumerate(smi_tokenizer(cano_prod_am)):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = re.match('.*:([0-9]+)]', token).group(1)
            if am in changed_atom_tags:
                nonreactive_mask.append(False)
                continue
        nonreactive_mask.append(True)

    if sum(nonreactive_mask) == len(nonreactive_mask):  # if the reaction center is not detected
        nonreactive_mask = [False] * len(nonreactive_mask)
    return nonreactive_mask
