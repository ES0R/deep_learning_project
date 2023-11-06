import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout, activation=activation)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout, activation=activation)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        # This is a PyTorch default initialization method for transformers.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Forward through the encoder
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Forward through the decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

# You can initialize the transformer like this:
# transformer = Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
