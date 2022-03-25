from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import math
import torch
import torch.nn as nn
import numpy as np

from retroformer.models.encoder import TransformerEncoder
from retroformer.models.decoder import TransformerDecoder
from retroformer.models.embedding import Embedding
from retroformer.models.module import MultiHeadedAttention


class RetroModel(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 vocab_size_src, vocab_size_tgt, shared_vocab, num_bonds=5, shared_encoder=False, src_pad_idx=1,
                 tgt_pad_idx=1):
        super(RetroModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.shared_vocab = shared_vocab
        self.shared_encoder = shared_encoder
        if shared_vocab:
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            self.embedding_src = self.embedding_tgt = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx)
        else:
            self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx)
            self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx)

        self.embedding_bond = nn.Linear(7, d_model)

        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(num_layers)])
        if shared_encoder:
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [MultiHeadedAttention(heads, d_model, dropout=dropout)
                 for _ in range(num_layers)])

        self.encoder = TransformerEncoder(num_layers=num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          embeddings_bond=self.embedding_bond,
                                          attn_modules=multihead_attn_modules_en)

        self.decoder = TransformerDecoder(num_layers=num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          self_attn_modules=multihead_attn_modules_de)

        self.atom_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())
        self.bond_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, bond=None, teacher_mask=None):
        encoder_out, edge_feature = self.encoder(src, bond)

        atom_rc_scores = self.atom_rc_identifier(encoder_out)
        bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None

        if teacher_mask is None:  # Naive Inference
            student_mask = self.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, student_mask.clone())
        else:  # Training
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, teacher_mask.clone())

        generative_scores = self.generator(decoder_out)

        return generative_scores, atom_rc_scores, bond_rc_scores, top_aligns

    @staticmethod
    def infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores=None):
        atom_rc_scores = atom_rc_scores.squeeze(2)
        if bond_rc_scores is not None:
            bond_rc_scores = bond_rc_scores.squeeze(1)
            bond_indicator = torch.zeros((bond.shape[0], bond.shape[1], bond.shape[2])).bool().to(bond.device)
            bond_indicator[bond.sum(-1) > 0] = (bond_rc_scores > 0.5)

            result = (~(bond_indicator.sum(dim=1).bool()) + ~(bond_indicator.sum(dim=2).bool()) +
                      (atom_rc_scores.transpose(0, 1) < 0.5)).transpose(0, 1)
        else:
            result = (atom_rc_scores.transpose(0, 1) < 0.5).transpose(0, 1)
        return result
