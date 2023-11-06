import torch
import torch.nn as nn
from models.backbone import build_backbone
from models.transformer import Transformer
from models.utils import MLP

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        
        # Prediction heads for class and bounding box
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.aux_loss = aux_loss

    def forward(self, samples):
        # Propagate samples through the backbone
        features, pos = self.backbone(samples)
        
        # Flatten the features and combine with positional encodings
        src, mask = features.decompose()
        hs = self.transformer(self.query_embed.weight, src, src_key_padding_mask=mask, pos=pos)
        
        # Prediction heads
        outputs_classes = self.class_embed(hs)
        outputs_coords = self.bbox_embed(hs).sigmoid()
        
        out = {'pred_logits': outputs_classes[-1], 'pred_boxes': outputs_coords[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes, outputs_coords)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_classes, outputs_coords):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_classes[:-1], outputs_coords[:-1])]

# You can initialize the DETR model like this:
# backbone = build_backbone(...)
# transformer = Transformer(...)
# detr = DETR(backbone, transformer, num_classes=91, num_queries=100)
