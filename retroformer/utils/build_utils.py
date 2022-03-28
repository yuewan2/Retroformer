from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from functools import partial
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from retroformer.dataset import RetroDataset
from retroformer.models.model import RetroModel


def load_checkpoint(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optim']
    step = checkpoint['step']
    step += 1
    return step, optimizer, model.to(args.device)


def build_model(args, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = RetroModel(
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
        shared_vocab=args.shared_vocab == 'True', shared_encoder=args.shared_encoder == 'True',
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)

    return model.to(args.device)


def build_iterator(args, train=True, sample=False, augment=False):
    if train:
        dataset = RetroDataset(mode='train', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True', sample=sample, augment=augment)
        dataset_val = RetroDataset(mode='val', data_folder=args.data_dir,
                                   intermediate_folder=args.intermediate_dir,
                                   known_class=args.known_class == 'True',
                                   shared_vocab=args.shared_vocab == 'True', sample=sample)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample,  # num_workers=8,
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos

    else:
        dataset = RetroDataset(mode='test', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True')
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset


def collate_fn(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, src_graph, tgt, alignment, nonreactive_mask = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([], device=device)

    # Graph structure with edge attributes
    new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, 7), dtype=torch.long)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)
    new_nonreactive_mask = anchor.new_ones((max_src_length, len(data)), dtype=torch.bool)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_nonreactive_mask[:, i][:len(nonreactive_mask[i])] = torch.BoolTensor(nonreactive_mask[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

        full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
        new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix

    return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph)


def accumulate_batch(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt, _, _, _ = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()

    new_context_alignment = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()
    new_nonreactive_mask = torch.ones((src_max_length, entry_count)).bool()

    # Graph packs:
    new_bond_matrix = torch.zeros((entry_count, src_max_length, src_max_length, 7)).long()
    new_src_graph_list = []

    for i in range(len(true_batch)):
        src, tgt, context_alignment, nonreactive_mask, graph_packs = true_batch[i]
        bond, src_graph = graph_packs
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_nonreactive_mask[:, batch_size * i: batch_size * (i + 1)][:nonreactive_mask.shape[0]] = nonreactive_mask
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_context_alignment[batch_size * i: batch_size * (i + 1), :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment

        new_bond_matrix[batch_size * i: batch_size * (i + 1), :bond.shape[1], :bond.shape[2]] = bond
        new_src_graph_list += src_graph

    return new_src, new_tgt, new_context_alignment, new_nonreactive_mask, \
           (new_bond_matrix, new_src_graph_list)