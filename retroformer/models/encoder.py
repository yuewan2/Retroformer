import torch
import torch.nn as nn
import torch.nn.functional as F

from retroformer.models.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, edge_feature, pair_indices):
        input_norm = self.layer_norm(inputs)
        context, attn, edge_feature_updated = self.self_attn(input_norm, input_norm, input_norm, mask=mask,
                                                             edge_feature=edge_feature,
                                                             pair_indices=pair_indices)

        out = self.dropout(context) + inputs
        if edge_feature is not None:
            edge_feature = self.layer_norm(edge_feature + edge_feature_updated)
        return self.feed_forward(out), attn, edge_feature

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, embeddings_bond, attn_modules):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.embeddings_bond = embeddings_bond
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)


    def forward(self, src, bond=None):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''
        global node_feature
        emb = self.embeddings(src)
        out = emb.transpose(0, 1).contiguous()

        if bond is not None:
            pair_indices = torch.where(bond.sum(-1) > 0)
            valid_bond = bond[bond.sum(-1) > 0]
            edge_feature = self.embeddings_bond(valid_bond.float())
        else:
            pair_indices, edge_feature = None, None

        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn, edge_feature = self.transformer[i](out, mask, edge_feature, pair_indices)

        out = self.layer_norm(out)
        out = out.transpose(0, 1).contiguous()
        edge_out = self.layer_norm(edge_feature) if edge_feature is not None else None
        return out, edge_out

