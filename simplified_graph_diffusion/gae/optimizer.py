# optimizer for gae
import torch
import torch.nn.modules.loss
import torch.nn.functional as F

def loss_function(preds, labels, norm, pos_weight):
    reconstruction_loss = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return reconstruction_loss