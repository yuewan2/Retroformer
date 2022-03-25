import math
import copy
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from retroformer.utils.smiles_utils import *


def scale(x):
    if (x != 0).sum() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())


def var(a):
    return a.clone().detach()
    # return torch.tensor(a, requires_grad=False)


def rvar(a, beam_size=10):
    if len(a.size()) == 3:
        return var(a.repeat(1, beam_size, 1))
    else:
        return var(a.repeat(1, beam_size))


def translate_batch_original(model, batch, eos_idx=3, beam_size=10, max_length=200,
                             invalid_token_indices=[], dataset=None):
    """
    :param inputs: tuple of (src, src_am, src_seg, tgt), tgt is only used to retrieve conditional reaction class token
    :param fixed_z: latent variable flag
    :param seed: latent variable flag
    :param target_mask_num: available only when generalize=False; constraint the amount of generated fragment = num of <MASK>
    :param sep_idx: target seperator '>>' index, only use when generalize=True; constraint the beam search from getting seperator too early
    :param prefix_sequence: list of prefix tokens, only use in customized template generation stage
    :return:
    """
    model.eval()
    src, tgt, gt_context_alignment, nonreactive_mask, graph_packs = batch
    bond, src_graph = graph_packs

    batch_size = src.shape[1]

    pred_tokens = src.new_ones((batch_size, beam_size, max_length + 1), dtype=torch.long)
    pred_scores = src.new_zeros((batch_size, beam_size), dtype=torch.float)
    pred_tokens[:, :, 0] = 2
    batch2finish = {i: False for i in range(batch_size)}

    # Encoder:
    with torch.no_grad():
        prior_encoder_out, edge_feature = model.encoder(src, bond)
        atom_rc_scores = model.atom_rc_identifier(prior_encoder_out)
        bond_rc_scores = model.bond_rc_identifier(edge_feature) if edge_feature is not None else None
        student_mask = model.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
        teacher_mask = nonreactive_mask

    # Get Student mask from graph search algorithm:
    src_repeat = rvar(src.data, beam_size=beam_size)
    memory_bank_repeat = rvar(prior_encoder_out.data, beam_size=beam_size)
    student_mask_repeat = rvar(student_mask.data, beam_size=beam_size)

    state_cache = {}
    for step in range(0, max_length):
        inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0, 1)

        with torch.no_grad():
            outputs, attn = model.decoder(src_repeat, inp, memory_bank_repeat, student_mask_repeat,
                                          state_cache=state_cache, step=step)
            scores = model.generator(outputs[-1])

        unbottle_scores = scores.view(beam_size, batch_size, -1)

        # Avoid invalid token:
        unbottle_scores[:, :, invalid_token_indices] = -1e25

        # Avoid token that end earily
        if step < 2:
            unbottle_scores[:, :, eos_idx] = -1e25

        # Beam Search:
        selected_indices = []
        for j in range(batch_size):
            prev_score = pred_scores[j].clone()
            batch_score = unbottle_scores[:, j]
            num_words = batch_score.size(1)
            # Get previous token to identify <eos>
            prev_token = pred_tokens[j, :, step]
            eos_index = prev_token.eq(eos_idx)
            # Prevent <eos> sequence to have children
            prev_score[eos_index] = -1e20

            if beam_size == eos_index.sum():  # all beam has finished
                pred_tokens[j, :, step + 1] = eos_idx
                batch2finish[j] = True
                selected_indices.append(torch.arange(beam_size, dtype=torch.long, device=src.device))
            else:
                beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                if step == 0:
                    flat_beam_scores = beam_scores[0].view(-1)
                else:
                    flat_beam_scores = beam_scores.view(-1)

                # Select the top-k highest accumulative scores
                k = beam_size - eos_index.sum().item()
                best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                # Freeze the tokens which has already finished
                frozed_tokens = pred_tokens[j][eos_index]
                if frozed_tokens.shape[0] > 0:
                    frozed_tokens[:, step + 1] = eos_idx
                frozed_scores = pred_scores[j][eos_index]

                # Update the rest of tokens
                origin_tokens = pred_tokens[j][best_scores_id // num_words]
                origin_tokens[:, step + 1] = best_scores_id % num_words

                updated_scores = torch.cat([best_scores, frozed_scores])
                updated_tokens = torch.cat([origin_tokens, frozed_tokens])

                pred_tokens[j] = updated_tokens
                pred_scores[j] = updated_scores

                if eos_index.sum() > 0:
                    tmp_indices = src.new_zeros(beam_size, dtype=torch.long)
                    tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                    selected_indices.append(tmp_indices)
                else:
                    selected_indices.append((best_scores_id // num_words))

            if dataset is not None:
                if j == 0:
                    hypos = [''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in updated_tokens]
                    print('[step {}]'.format(step))
                    for hypo in hypos:
                        print(hypo)
                    # print(hypos[0])
                    print('------------------`')

        if selected_indices:
            reorder_state_cache(state_cache, selected_indices)

        if sum(batch2finish.values()) == len(batch2finish):
            break

    # (Sorting is done in explain_batch)
    return pred_tokens, pred_scores


def translate_batch_stepwise(model, batch, eos_idx=3, beam_size=10, max_length=200, invalid_token_indices=[], T=5,
                             alpha_atom=0.0001, alpha_bond=0.05, percent_aa=0, percent_ab=0, beta=0.5, k=4,
                             verbose=False, factor_func=None, use_template=False, reconstruct_func=None, rc_path=''):
    '''
    :param inputs: tuple of (src, src_am, src_seg, tgt), tgt is only used to retrieve conditional reaction class token
    :param fixed_z: latent variable flag
    :param seed: latent variable flag
    :param target_mask_num: available only when generalize=False; constraint the amount of generated fragment = num of <MASK>
    :param sep_idx: target seperator '>>' index, only use when generalize=True; constraint the beam search from getting seperator too early
    :param prefix_sequence: list of prefix tokens, only use in customized template generation stage
    :return:
    '''

    model.eval()

    src, tgt, gt_context_alignment, nonreactive_mask, graph_packs = batch
    teacher_mask = nonreactive_mask
    bond, src_graph = graph_packs

    # Encoder:
    with torch.no_grad():
        prior_encoder_out, edge_feature = model.encoder(src, bond)
        atom_rc_scores = model.atom_rc_identifier[1](model.atom_rc_identifier[0](prior_encoder_out) / T)
        bond_rc_scores = model.bond_rc_identifier[1](model.bond_rc_identifier[0](edge_feature) / T)

    raw_predicts = model.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)

    # Get Student mask from subgraph scoring:
    if not use_template:
        _, predicts = batch_infer_reaction_center(atom_rc_scores, bond_rc_scores, graph_packs,
                                                  alpha_atom=alpha_atom, alpha_bond=alpha_bond, beta=beta,
                                                  percent_aa=percent_aa, percent_ab=percent_ab,
                                                  verbose=verbose, k=k, factor_func=factor_func,
                                                  num_removal=5, max_count=25)

    # Get Student mask from template scoring:
    else:
        assert rc_path and reconstruct_func is not None
        with open(rc_path, 'rb') as f:
            rt2reaction_center = pickle.load(f)

        pair_indices = torch.where(bond.sum(-1) > 0)
        batch_bond_scores = torch.zeros(src.shape[1], src.shape[0], src.shape[0]).to(src.device)
        batch_bond_scores[pair_indices] = bond_rc_scores.view(-1)
        batch_atom_scores = atom_rc_scores.squeeze(2).transpose(0, 1)

        predicts = []
        for batch_i in range(src.shape[1]):
            src_tokens = reconstruct_func(src[:, batch_i], src=True)
            rt_token, src_tokens = src_tokens[0], src_tokens[1:]
            for i in range(len(src_tokens)):
                src_tokens[i] = add_mapping(src_tokens[i], map_num=i)
            src_smiles = ''.join(src_tokens)
            # print(src_smiles)
            atom_scores = (batch_atom_scores[batch_i][1:]).cpu().numpy()
            bond_scores = (batch_bond_scores[batch_i][1:, 1:]).cpu().numpy()
            adjacency_matrix = src_graph[batch_i].adjacency_matrix
            full_adjacency_matrix = src_graph[batch_i].full_adjacency_tensor.sum(-1)
            graph_pack = (atom_scores, bond_scores, adjacency_matrix, full_adjacency_matrix)

            cc_trace_with_score_template = get_reaction_centers_from_template(src_smiles, graph_pack,
                                                                              rt2reaction_center[rt_token])
            if len(cc_trace_with_score_template):
                predict_rc_template = select_diverse_candidate(cc_trace_with_score_template, diverse_k=k)
            else:
                predict_rc_template = []
            predicts.append(predict_rc_template)

    # Remake batched data:
    new_src, new_prior_encoder_out, new_student_mask = [], [], []
    for batch_i, predict in enumerate(predicts):
        if len(predict):
            new_src += [src[:, batch_i]] * len(predict)
            new_prior_encoder_out += [prior_encoder_out[:, batch_i]] * len(predict)
            for rc, score in predict:
                rc_indices = np.array(rc) + 1  # padding=1
                new_s_mask = torch.BoolTensor(src.shape[0]).fill_(True).to(src.device)
                new_s_mask[rc_indices] = False
                new_student_mask.append(new_s_mask)
        else:
            new_src += [src[:, batch_i]] * 1
            new_prior_encoder_out += [prior_encoder_out[:, batch_i]] * 1
            predicts[batch_i] = [(tuple(list(torch.where(~raw_predicts[:, batch_i])[0].cpu().numpy() - 1)), 0)]
            new_student_mask.append(raw_predicts[:, batch_i])

    new_src = torch.stack(new_src).transpose(0, 1)
    new_prior_encoder_out = torch.stack(new_prior_encoder_out).transpose(0, 1)
    new_student_mask = torch.stack(new_student_mask).transpose(0, 1)
    batch_size = new_src.shape[1]

    # Decoder:
    src_repeat = rvar(new_src.data, beam_size=beam_size)
    memory_bank_repeat = rvar(new_prior_encoder_out.data, beam_size=beam_size)
    student_mask_repeat = rvar(new_student_mask.data, beam_size=beam_size)

    # Initialize
    pred_tokens = torch.ones((batch_size, beam_size, max_length + 1)).long().to(src.device)
    pred_tokens[:, :, 0] = 2
    pred_scores = torch.zeros((batch_size, beam_size)).to(src.device)
    batch2finish = {i: False for i in range(batch_size)}

    state_cache = {}
    for step in range(0, max_length):
        torch.cuda.empty_cache()
        inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0, 1).to(
            src.device)

        with torch.no_grad():
            outputs, attn = model.decoder(src_repeat, inp, memory_bank_repeat, student_mask_repeat,
                                          state_cache=state_cache, step=step)
            scores = model.generator(outputs[-1])

        unbottle_scores = scores.view(beam_size, batch_size, -1)

        # Avoid invalid token:
        unbottle_scores[:, :, invalid_token_indices] = -1e25

        # Avoid token that end earily
        if step < 2:
            unbottle_scores[:, :, eos_idx] = -1e25

        # Beam Search:
        selected_indices = []
        for j in range(batch_size):
            prev_score = copy.deepcopy(pred_scores[j])
            batch_score = unbottle_scores[:, j]
            num_words = batch_score.size(1)
            # Get previous token to identify <eos>
            prev_token = pred_tokens[j, :, step]
            eos_index = prev_token.eq(eos_idx)
            # Prevent <eos> sequence to have children
            prev_score[eos_index] = -1e20

            if beam_size == eos_index.sum():  # all beam has finished
                pred_tokens[j, :, step + 1] = eos_idx
                batch2finish[j] = True
                selected_indices.append(torch.LongTensor(np.arange(beam_size)).to(src.device))
            else:
                beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                if step == 0:
                    flat_beam_scores = beam_scores[0].view(-1)
                else:
                    flat_beam_scores = beam_scores.view(-1)

                # Select the top-k highest accumulative scores
                k = beam_size - eos_index.sum()
                best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                # Freeze the tokens which has already finished
                frozed_tokens = pred_tokens[j][eos_index]
                if frozed_tokens.shape[0] > 0:
                    frozed_tokens[:, step + 1] = eos_idx
                frozed_scores = pred_scores[j][eos_index]

                # Update the rest of tokens
                origin_tokens = pred_tokens[j][best_scores_id // num_words]
                origin_tokens[:, step + 1] = best_scores_id % num_words

                updated_scores = torch.cat([best_scores, frozed_scores])
                updated_tokens = torch.cat([origin_tokens, frozed_tokens])

                pred_tokens[j] = updated_tokens
                pred_scores[j] = updated_scores

                if eos_index.sum() > 0:
                    tmp_indices = torch.zeros(beam_size).long().to(src.device)
                    tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                    selected_indices.append(tmp_indices)
                else:
                    selected_indices.append((best_scores_id // num_words))

        if selected_indices:
            reorder_state_cache(state_cache, selected_indices)

        if sum(batch2finish.values()) == len(batch2finish):
            break

    # (Sorting is done later)
    return pred_tokens, pred_scores, predicts


def batch_infer_reaction_center(atom_rc_scores, bond_rc_scores, graph_packs,
                                alpha_atom=0.001, alpha_bond=0.001, beta=0.65,
                                percent_aa=0, percent_ab=0, max_count=-1, verbose=False,
                                k=3, num_removal=5,
                                factor_func=None):
    """Batch reaction center search"""
    raw_predicts, predicts = [], []
    bond, src_graph = graph_packs

    pair_indices = torch.where(bond.sum(-1) > 0)
    batch_bond_scores = torch.zeros(atom_rc_scores.shape[1], atom_rc_scores.shape[0], atom_rc_scores.shape[0]).to(
        bond.device)
    batch_bond_scores[pair_indices] = bond_rc_scores.view(-1)
    batch_atom_scores = atom_rc_scores.squeeze(2).transpose(0, 1)

    for batch_i in range(bond.shape[0]):
        atom_scores = (batch_atom_scores[batch_i][1:]).cpu().numpy()
        bond_scores = (batch_bond_scores[batch_i][1:, 1:]).cpu().numpy()

        if percent_aa > 0:
            alpha_atom = np.percentile(atom_scores, percent_aa)
            if verbose:
                print('Computed alpha_atom = {}'.format(alpha_atom))
        if percent_ab > 0:
            alpha_bond = np.percentile(bond_scores[bond_scores > 0], percent_ab)
            if verbose:
                print('Computed alpha_bond = {}'.format(alpha_bond))

        atom_scores[bond_scores.sum(-1) == 0] = 0

        if verbose:
            print('-------- Atom Scores --------')
            print(atom_scores)
            print('-------- Bond Scores --------')
            print(bond_scores[bond_scores > 0])

        adjacency_matrix = src_graph[batch_i].adjacency_matrix
        full_adjacency_matrix = src_graph[batch_i].full_adjacency_tensor.sum(-1)
        graph_pack = (atom_scores, bond_scores, adjacency_matrix, full_adjacency_matrix)

        raw_predicts.append(tuple(np.argwhere(atom_scores > 0.5).flatten()))

        global_min_count = min(max(2, (atom_scores > beta).sum()), max_count)

        # Get Parent Substructures
        cc_trace_candidates = []
        cc_trace_parents = []
        visited = [False] * len(atom_scores)
        for head_i in range(len(adjacency_matrix)):
            if atom_scores[head_i] > alpha_atom and not visited[head_i]:
                cc_trace = dfs_cc_atom([head_i], head_i, visited, graph_pack, alpha_atom)
                # cc_trace = dfs_cc([head_i], head_i, visited, graph_pack, alpha_atom, alpha_bond)
                if len(cc_trace) > 1:
                    cc_trace_parents.append(cc_trace)
                    # cc_trace_candidates.append(cc_trace)

        # Get Children Substructures
        visited = [False] * len(atom_scores)
        for parent_i, cc_trace_parent in enumerate(cc_trace_parents):
            child_found = False
            for head_i in cc_trace_parent:
                if atom_scores[head_i] > alpha_atom and not visited[head_i]:
                    visited_copy = visited[:]
                    cc_trace = dfs_cc_bond([head_i], head_i, visited, graph_pack, cc_trace_parent, alpha_bond)

                    if len(cc_trace) > 1:
                        if 0 < max_count < len(cc_trace):
                            # print('Warning: size above limits.')
                            cc_trace = dfs_cc_bond([head_i], head_i, visited_copy, graph_pack, cc_trace_parent, 0.3)
                            visited = visited_copy
                        cc_trace_candidates.append(cc_trace)
                        child_found = True

            if not child_found and 6 > len(cc_trace_parent) > 3:
                cc_trace_candidates.append(cc_trace_parent)

        if verbose:
            print('----- Substructure Candidates -----')
            for cc_trace in cc_trace_candidates:
                print(cc_trace)
            print('-----------------------------------')

        rc2score = {}
        pred_reaction_centers = []
        for cc_trace in cc_trace_candidates:
            if len(cc_trace) < global_min_count:
                if len(cc_trace) > 1:
                    factor = 1 if factor_func is None else factor_func(len(cc_trace))
                    cc_score_total = get_cc_score(cc_trace, graph_pack)
                    pred_reaction_centers.append(
                        [(tuple(sorted(cc_trace)), cc_score_total / (factor * get_norm(cc_trace, graph_pack)))])
                continue

            cc_score_total = get_cc_score(cc_trace, graph_pack)
            # local_min_count = max(min(global_min_count, 5), np.sum(atom_scores[cc_trace] > 0.6), 2)
            local_min_count = max(2, np.sum(atom_scores[cc_trace] > 0.5))
            if verbose:
                print(global_min_count, local_min_count)
                print('Parent:', cc_trace)

            sub_cc_trace_candidates = get_subgraphs_by_trim(cc_trace, cc_score_total, graph_pack,
                                                            min_count=local_min_count, max_count=40,
                                                            num_removal=num_removal,
                                                            verbose=False)

            assert sub_cc_trace_candidates
            sub_pred_candidates = []
            for rc_cand in sub_cc_trace_candidates:
                factor = 1 if factor_func is None else factor_func(len(rc_cand))
                sub_pred_candidates.append(
                    (rc_cand, get_cc_score(rc_cand, graph_pack) / (factor * get_norm(rc_cand, graph_pack))))

            sub_pred_candidates = sorted(sub_pred_candidates, key=lambda x: -x[1])
            pred_reaction_centers.append(sub_pred_candidates)

            if verbose:
                print('Children:')
                for sub_cand in sub_pred_candidates:
                    print(round(sub_cand[1], 4), sub_cand[0],
                          set_overlap(sub_pred_candidates[0][0], sub_cand[0]))  # get_norm(sub_cand[0], graph_pack)
                print()

            pred_rcs = pred_reaction_centers[-1]
            rc2score.update({rc: score for rc, score in pred_rcs})

        diverse_pred_reaction_centers = []
        for pred_rcs in pred_reaction_centers:
            top_k_diverse_pred_rcs = select_diverse_candidate(pred_rcs, diverse_k=3)
            diverse_pred_reaction_centers += top_k_diverse_pred_rcs

        diverse_pred_reaction_centers = sorted(diverse_pred_reaction_centers, key=lambda x: -x[1])
        predict = []
        for pred_rcs in diverse_pred_reaction_centers:
            if pred_rcs[1] > np.log(0.3):
                predict.append(pred_rcs)
        # predict = diverse_pred_reaction_centers
        predicts.append(predict[:k])

    return raw_predicts, predicts


def reorder_state_cache(state_cache, selected_indices):
    """Reorder state_cache of the decoder
    params state_cache: list of indices
    params selected_indices: size (batch_size x beam_size)
    """
    batch_size, beam_size = len(selected_indices), len(selected_indices[0])
    indices_mapping = torch.arange(batch_size * beam_size,
                                   device=selected_indices[0].device).reshape(beam_size, batch_size).transpose(0, 1)
    reorder_indices = []
    for batch_i, indices in enumerate(selected_indices):
        reorder_indices.append(indices_mapping[batch_i, indices])
    reorder_indices = torch.stack(reorder_indices, dim=1).view(-1)

    new_state_cache = []
    for key in state_cache:
        if isinstance(state_cache[key], dict):
            for subkey in state_cache[key]:
                state_cache[key][subkey] = state_cache[key][subkey][reorder_indices]

        elif isinstance(state_cache[key], torch.Tensor):
            state_cache[key] = state_cache[key][reorder_indices]
        else:
            raise Exception
