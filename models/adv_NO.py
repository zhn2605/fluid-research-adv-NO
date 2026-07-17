import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    # PatchGAN-style critic: velocity field [B, 2, H, W] -> grid of realness
    # logits. Judges local patches, so it pressures the small-scale structure
    # that L1/L2 ignores.

    def __init__(self, input_channels: int = 2, base_width: int = 64):
        super().__init__()
        # 

    def forward(self, x):
        # returns patch logits [B, 1, h', w']
        raise NotImplementedError


class AdvNOLoss(nn.Module):
    # similar to PITALoss's interface (forward(pred_seq, true_seq) -> total, parts)
    # so it can slot into train_recursive behind a use_adv flag. The difference:
    # the critic itself has trainable weights that must be optimized AGAINST the
    # generator, so training needs two optimizers updated alternately per batch:
    #   1. opt_G on model.parameters()      minimizing generator_loss
    #   2. opt_D on loss.D.parameters()     minimizing discriminator_loss

    def __init__(self, beta_adv: float = 0.1, input_channels: int = 2):
        super().__init__()
        self.D = Discriminator(input_channels)
        self.beta_adv = beta_adv  # paper weights the adversarial term at 0.1

    @staticmethod
    def _frames(seq):
        # [B, T, C, H, W] -> [B*T, C, H, W]; the critic scores single frames
        raise NotImplementedError

    def _relativistic(self, real_logits, fake_logits):
        # RaGAN: score each sample relative to the other population's mean,
        #   real vs fake: real_logits - fake_logits.mean()
        #   fake vs real: fake_logits - real_logits.mean()
        raise NotImplementedError

    def generator_loss(self, pred_seq, true_seq):
        # TODO: L1 reconstruction term (keeps large scales anchored to truth).
        # TODO: adversarial term: BCE-with-logits on the relativistic scores,
        #       labels flipped so the generator is rewarded for fooling D.
        #       D participates in the graph here but only opt_G steps.
        raise NotImplementedError

    def discriminator_loss(self, pred_seq, true_seq):
        # TODO: same RaGAN objective, but now D is the only participant 
        raise NotImplementedError

    def forward(self, pred_seq, true_seq):
        return self.generator_loss(pred_seq, true_seq)
