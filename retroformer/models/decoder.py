import torch
import torch.nn as nn
import numpy as np

from retroformer.models.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn, context_attn):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.context_attn = context_attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(5000)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                nonreactive_mask_input=None, layer_input=None, layer_cache=None):
        # inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
        # memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
        # src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
        # infer_decision_input (`LongTensor`): `[batch_size x tgt_len]`
        # nonreactive_mask_input (`BoolTensor`): `[batch_size x src_len]`
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)

        # Self-attention:
        all_input = input_norm
        if layer_input is not None:
            all_input = torch.cat((layer_input, input_norm), dim=1)
            dec_mask = None
        query, self_attn, _ = self.self_attn(all_input, all_input, input_norm,
                                             mask=dec_mask,
                                             type="self",
                                             layer_cache=layer_cache)
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        # Context-attention:
        mid, context_attn, _ = self.context_attn(memory_bank, memory_bank, query_norm,
                                                 mask=src_pad_mask,
                                                 additional_mask=nonreactive_mask_input,
                                                 type="context",
                                                 layer_cache=layer_cache)
        output = self.feed_forward(self.drop(mid) + query)

        return output, context_attn, all_input

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, self_attn_modules):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings

        context_attn_modules = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(num_layers)])

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout, self_attn_modules[i], context_attn_modules[i])
             for i in range(num_layers)])

        self.layer_norm_0 = LayerNorm(d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, tgt, memory_bank, nonreactive_mask=None, state_cache=None, step=None):
        '''
        :param src:
        :param tgt:
        :param memory_bank:
        :param nonreactive_mask: mask corresponding to reaction center identification from encoder
        :param infer_label: only occur in training for teacher's forcing; during inference, infer_label is the infer_decision.
        :param state_cache:
        :param step:
        :return:
        '''
        if nonreactive_mask is not None:
            nonreactive_mask[0] = False     # allow attention to the initial src token

        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Initialize return variables.
        outputs = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            tgt_batch, tgt_len = tgt_words.size()

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        # assume src padding idx and tgt padding idx are the same
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        nonreactive_mask_input = nonreactive_mask.transpose(0, 1) if nonreactive_mask is not None else None
        top_context_attns = []
        for i in range(self.num_layers):
            layer_input = None
            layer_cache = {'self_keys': None,
                           'self_values': None,
                           'memory_keys': None,
                           'memory_values': None}
            if state_cache is not None:
                layer_cache = state_cache.get('layer_cache_{}'.format(i), layer_cache)
                layer_input = state_cache.get('layer_input_{}'.format(i), layer_input)

            output, top_context_attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    layer_input=layer_input,
                    layer_cache=layer_cache,
                    nonreactive_mask_input=nonreactive_mask_input)

            top_context_attns.append(top_context_attn)
            if state_cache is not None:
                state_cache['layer_cache_{}'.format(i)] = layer_cache
                state_cache['layer_input_{}'.format(i)] = all_input


        output = self.layer_norm(output)
        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        return outputs, top_context_attns
