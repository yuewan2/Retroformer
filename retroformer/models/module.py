import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from OpenNMT
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.edge_project = nn.Sequential(nn.Linear(model_dim, model_dim),
                                          SSP(),
                                          nn.Linear(model_dim, model_dim // 2))
        self.edge_update = nn.Sequential(nn.Linear(model_dim * 2, model_dim),
                                         SSP(),
                                         nn.Linear(model_dim, model_dim))

    def forward(self, key, value, query, mask=None, additional_mask=None,
                layer_cache=None, type=None, edge_feature=None, pair_indices=None):

        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        global query_projected, key_shaped, value_shaped
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query_projected, key_projected, value_projected = self.linear_query(query), \
                                                                  self.linear_keys(query), \
                                                                  self.linear_values(query)

                key_shaped = shape(key_projected)
                value_shaped = shape(value_projected)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key_shaped = torch.cat(
                            (layer_cache["self_keys"].to(device), key_shaped),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value_shaped = torch.cat(
                            (layer_cache["self_values"].to(device), value_shaped),
                            dim=2)
                    layer_cache["self_keys"] = key_shaped
                    layer_cache["self_values"] = value_shaped
            elif type == "context":
                query_projected = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key_projected, value_projected = self.linear_keys(key), \
                                                         self.linear_values(value)
                        key_shaped = shape(key_projected)
                        value_shaped = shape(value_projected)
                    else:
                        key_shaped, value_shaped = layer_cache["memory_keys"], \
                                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key_shaped
                    layer_cache["memory_values"] = value_shaped
                else:
                    key_projected, value_projected = self.linear_keys(key), \
                                                     self.linear_values(value)
                    key_shaped = shape(key_projected)
                    value_shaped = shape(value_projected)
        else:
            key_projected = self.linear_keys(key)
            value_projected = self.linear_values(value)
            query_projected = self.linear_query(query)
            key_shaped = shape(key_projected)
            value_shaped = shape(value_projected)

        query_shaped = shape(query_projected)
        key_len = key_shaped.size(2)
        query_len = query_shaped.size(2)

        # 2) Calculate and scale scores (local and global attention heads).
        # 3) Apply attention dropout and compute context vectors.
        # Local-Global Decoder:
        if edge_feature is None and additional_mask is not None:
            query_shaped = query_shaped / math.sqrt(dim_per_head)
            query_shaped_global, query_shaped_local = query_shaped[:, :head_count // 2], query_shaped[:,
                                                                                         head_count // 2:]
            key_shaped_global, key_shaped_local = key_shaped[:, :head_count // 2], key_shaped[:, head_count // 2:]
            value_shaped_global, value_shaped_local = value_shaped[:, :head_count // 2], value_shaped[:,
                                                                                         head_count // 2:]

            # Global:
            score_global = torch.matmul(query_shaped_global, key_shaped_global.transpose(2, 3))
            top_score = score_global.view(batch_size, score_global.shape[1],
                                          query_len, key_len)[:, 0, :, :].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(score_global).clone()
                score_global = score_global.masked_fill(mask, -1e18)
            attn = self.softmax(score_global)
            drop_attn = self.dropout(attn)
            global_context = torch.matmul(drop_attn, value_shaped_global)

            # Local:
            score_local = torch.matmul(query_shaped_local, key_shaped_local.transpose(2, 3))
            if additional_mask is not None:
                additional_mask = additional_mask.unsqueeze(1).unsqueeze(2).expand_as(score_local).clone()
                score_local = score_local.masked_fill(additional_mask, -1e18)
            attn = self.softmax(score_local)
            drop_attn = self.dropout(attn)
            local_context = torch.matmul(drop_attn, value_shaped_local)

            # Merge:
            context = torch.cat([global_context, local_context], dim=1)
            context = unshape(context)

        # Local-Global Encoder:
        elif edge_feature is not None:
            # Local: (node update)
            edge_feature_shaped = self.edge_project(edge_feature).view(-1, head_count // 2, dim_per_head)
            key_shaped_local = key_shaped[pair_indices[0], head_count // 2:, pair_indices[2]]
            query_shaped_local = query_shaped[pair_indices[0], head_count // 2:, pair_indices[1]]
            value_shaped_local = value_shaped[:, head_count // 2:]

            key_shaped_local = key_shaped_local * edge_feature_shaped
            query_shaped_local = query_shaped_local / math.sqrt(dim_per_head)

            scores_local = torch.matmul(query_shaped_local.unsqueeze(2),
                                        key_shaped_local.unsqueeze(3)).view(edge_feature.shape[0], head_count // 2)

            score_expand_local = scores_local.new_full(
                (value.shape[0], value.shape[1], value.shape[1], head_count // 2), -float('inf'))
            score_expand_local[pair_indices] = scores_local
            score_expand_local = score_expand_local.transpose(1, 3).transpose(2, 3)

            attn_local = self.softmax(score_expand_local)
            attn_local = attn_local.masked_fill(score_expand_local < -10000, 0)
            drop_attn_local = self.dropout(attn_local)
            local_context = torch.matmul(drop_attn_local, value_shaped_local)

            # Global：
            query_shaped_global = query_shaped[:, :head_count // 2]
            key_shaped_global = key_shaped[:, :head_count // 2]
            value_shaped_global = value_shaped[:, :head_count // 2]

            query_shaped_global = query_shaped_global / math.sqrt(dim_per_head)
            score_global = torch.matmul(query_shaped_global, key_shaped_global.transpose(2, 3))
            top_score = score_global.view(batch_size, score_global.shape[1],
                                          query_len, key_len)[:, 0, :, :].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(score_global).clone()
                score_global = score_global.masked_fill(mask, -1e18)

            attn = self.softmax(score_global)
            drop_attn = self.dropout(attn)
            global_context = torch.matmul(drop_attn, value_shaped_global)

            # Merge:
            context = torch.cat([global_context, local_context], dim=1)
            context = unshape(context)

        # Original Encoder/Decoder/Decoder self-attention:
        else:
            query_shaped = query_shaped / math.sqrt(dim_per_head)
            scores = torch.matmul(query_shaped, key_shaped.transpose(2, 3))
            top_score = scores.view(batch_size, scores.shape[1],
                                    query_len, key_len)[:, 0, :, :].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(scores).clone()
                if additional_mask is not None:  # Local head of decoder:
                    additional_mask = additional_mask.unsqueeze(1).expand((batch_size, head_count // 2,
                                                                           query_len, key_len))
                    mask[:, mask.shape[1] // 2:] = additional_mask
                scores = scores.masked_fill(mask, -1e18)
            attn = self.softmax(scores)
            drop_attn = self.dropout(attn)
            context = torch.matmul(drop_attn, value_shaped)
            context = unshape(context)

        output = self.final_linear(context)

        # 4) Edge update (only in encoder)
        if edge_feature is not None:
            node_feature_updated = output
            node_features = torch.cat(
                [node_feature_updated[pair_indices[0], pair_indices[1]],
                 node_feature_updated[pair_indices[0], pair_indices[2]]], dim=-1)
            edge_feature_updated = self.edge_update(node_features)

            return output, top_score, edge_feature_updated
        else:
            return output, top_score, None


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
