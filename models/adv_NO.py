# adv-NO (arXiv:2509.08752): discriminator, loss, EMA and spectrum metric.
# The generator is whatever model the benchmark trains (models/UNet.py).

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint


def _free_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(err):
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


# measured memory per critic/VGG frame, per pixel (~92 MiB/frame at 133x89)
_BYTES_PER_FRAME_PER_PIXEL = 8200


class FeatureExtractor(nn.Module):
    # frozen VGG-19 features, each velocity channel fed as a grayscale image
    # use_checkpoint saves memory by recomputing activations during backward

    def __init__(self, layers=(0, 5, 10, 19, 28), use_checkpoint=True):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.layers = tuple(layers)
        self.use_checkpoint = use_checkpoint
        self.net = nn.Sequential(*[vgg.features[i] for i in range(max(layers) + 1)])
        # inplace ReLU breaks checkpointing
        for m in self.net.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # always stay in eval mode
        return super().train(False)

    def forward(self, x):
        # x - [N, C, H, W] velocity frames -> features of [N*C, 3, H, W]
        n, c, h, w = x.shape
        img = x.reshape(n * c, 1, h, w).repeat(1, 3, 1, 1)
        features = []
        start = 0
        for end in self.layers:
            segment = self.net[start:end + 1]
            if self.use_checkpoint and img.requires_grad:
                img = checkpoint(segment, img, use_reentrant=False)
            else:
                img = segment(img)
            features.append(img)
            start = end + 1
        return features


class UNetDiscriminatorSN(nn.Module):
    # Real-ESRGAN style critic: [N, C, H, W] -> per-pixel logits [N, 1, H, W]

    def __init__(self, num_in_ch=2, num_feat=64, skip_connection=True):
        super().__init__()
        norm = spectral_norm
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    @staticmethod
    def _up(x, like):
        # 133x89 isn't divisible by 8, so match the skip's size
        return F.interpolate(x, size=like.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x4 = F.leaky_relu(self.conv4(self._up(x3, x2)), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x5 = F.leaky_relu(self.conv5(self._up(x4, x1)), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x6 = F.leaky_relu(self.conv6(self._up(x5, x0)), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


class EMA:
    # exponential moving average of the weights

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = ((1.0 - self.decay) * param.data
                                     + self.decay * self.shadow[name]).clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def energy_spectrum_2d(u):
    # u - [N, C, H, W] -> radially binned energy spectrum [N, n_bins]
    n, c, h, w = u.shape
    uh = torch.fft.fft2(u) / (h * w)
    energy = 0.5 * (uh.real ** 2 + uh.imag ** 2).sum(dim=1)  # [N, H, W]

    ki = torch.fft.fftfreq(h, d=1.0 / h, device=u.device)  # integer wavenumber indices
    kj = torch.fft.fftfreq(w, d=1.0 / w, device=u.device)
    kmag = torch.sqrt(ki[:, None] ** 2 + kj[None, :] ** 2)
    bins = torch.round(kmag).to(torch.int64).flatten()  # [H*W]
    n_bins = int(bins.max().item()) + 1

    spectrum = torch.zeros(n, n_bins, device=u.device, dtype=energy.dtype)
    spectrum.scatter_add_(1, bins.unsqueeze(0).expand(n, -1), energy.reshape(n, -1))
    return spectrum


def spectrum_error(pred_seq, true_seq, kmax=None, eps=1e-12):
    # mean (log E_true - log E_pred)^2 over bins 1..kmax, used for model selection
    with torch.no_grad():
        p = pred_seq.reshape(-1, *pred_seq.shape[-3:])
        t = true_seq.reshape(-1, *true_seq.shape[-3:])
        sp = energy_spectrum_2d(p)
        st = energy_spectrum_2d(t)
        if kmax is None:
            # 3/4 of the shorter axis' Nyquist
            kmax = int(min(p.shape[-2], p.shape[-1]) // 2 * 0.75)
        kmax = min(kmax, sp.shape[1] - 1)
        err = (torch.log(st[:, 1:kmax + 1] + eps) - torch.log(sp[:, 1:kmax + 1] + eps)) ** 2
        return err.mean().item()


class AdvNOLoss(nn.Module):
    # generator loss (paper Eq. 8): L1 + VGG perceptual + beta * RaGAN
    # same interface as PITALoss; D needs its own optimizer
    # before adversarial_enabled only the L1 term is used

    def __init__(self, input_channels=2, beta_adv=0.1, use_perceptual=True,
                 perceptual_weights=(0.1, 0.1, 1.0, 1.0, 1.0), max_disc_frames=16,
                 min_disc_frames=2, autoscale_on_oom=True, vgg_checkpoint=True,
                 fit_to_free_vram=True, vram_safety=0.5):
        super().__init__()
        self.D = UNetDiscriminatorSN(input_channels)
        self.beta_adv = beta_adv
        self.use_perceptual = use_perceptual
        self.perceptual_weights = tuple(perceptual_weights)
        # D and VGG only see max_disc_frames random frames per step (memory)
        self.max_disc_frames = max_disc_frames
        self.min_disc_frames = min_disc_frames
        self.autoscale_on_oom = autoscale_on_oom
        self.fit_to_free_vram = fit_to_free_vram
        self.vram_safety = vram_safety
        self._budget_fitted = False
        self._adversarial_enabled = False
        self.vgg = FeatureExtractor(use_checkpoint=vgg_checkpoint) if use_perceptual else None
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    @property
    def adversarial_enabled(self):
        return self._adversarial_enabled

    @adversarial_enabled.setter
    def adversarial_enabled(self, value):
        # free cached memory before stage 2 starts
        value = bool(value)
        if value and not getattr(self, "_adversarial_enabled", False):
            _free_cuda_cache()
            self._budget_fitted = False
        self._adversarial_enabled = value

    def _fit_frame_budget(self, h, w):
        # lower max_disc_frames to fit free VRAM (called after the rollout is allocated)
        self._budget_fitted = True
        if not (self.fit_to_free_vram and torch.cuda.is_available()):
            return
        free, _ = torch.cuda.mem_get_info()
        per_frame = _BYTES_PER_FRAME_PER_PIXEL * h * w
        allowed = int(free * self.vram_safety / max(per_frame, 1))
        allowed = max(self.min_disc_frames, min(self.max_disc_frames, allowed))
        if allowed < self.max_disc_frames:
            print(f"[adv-NO] {free / 2**20:.0f} MiB VRAM free entering stage 2; "
                  f"reducing max_disc_frames {self.max_disc_frames} -> {allowed} "
                  f"(~{per_frame / 2**20:.0f} MiB per frame)")
            if allowed < 8:
                print(f"[adv-NO] WARNING: only {allowed} critic frames per step, "
                      f"consider lowering batch_size")
            self.max_disc_frames = allowed

    def _retry_on_oom(self, fn, *args):
        # on OOM halve max_disc_frames and retry
        while True:
            try:
                return fn(*args)
            except (torch.OutOfMemoryError, RuntimeError) as err:
                if not (self.autoscale_on_oom and _is_oom(err)):
                    raise
                if self.max_disc_frames <= self.min_disc_frames:
                    raise
                new = max(self.min_disc_frames, self.max_disc_frames // 2)
                print(f"[adv-NO] OOM, max_disc_frames {self.max_disc_frames} -> {new}")
                self.max_disc_frames = new
                _free_cuda_cache()

    @staticmethod
    def _frames(seq):
        # [B, T, C, H, W] -> [B*T, C, H, W]
        return seq.reshape(-1, *seq.shape[-3:])

    def _subsample(self, pred_f, true_f):
        # same frames for pred and true
        n = pred_f.shape[0]
        if n <= self.max_disc_frames:
            return pred_f, true_f
        idx = torch.randperm(n, device=pred_f.device)[:self.max_disc_frames]
        return pred_f[idx], true_f[idx]

    def _ragan(self, first_logits, second_logits):
        # relativistic average GAN loss, first = real label, second = fake
        valid = torch.ones_like(first_logits)
        fake = torch.zeros_like(second_logits)
        return (self.bce(first_logits - second_logits.mean(0, keepdim=True), valid)
                + self.bce(second_logits - first_logits.mean(0, keepdim=True), fake)) / 2

    def _perceptual(self, pred_f, true_f):
        gen_features = self.vgg(pred_f)
        # no gradient needed for the target
        with torch.no_grad():
            real_features = self.vgg(true_f)
        return sum(self.l1(g, r) * w for g, r, w
                   in zip(gen_features, real_features, self.perceptual_weights))

    def _generator_loss(self, pred_seq, true_seq):
        # L1 over the full rollout
        loss_pixel = self.l1(pred_seq, true_seq)
        zero = torch.zeros((), device=pred_seq.device)
        parts = {'L_pixel': loss_pixel, 'L_percep': zero, 'L_adv': zero}

        if not self.adversarial_enabled:
            return loss_pixel, parts

        if not self._budget_fitted:
            self._fit_frame_budget(pred_seq.shape[-2], pred_seq.shape[-1])

        pred_f, true_f = self._subsample(self._frames(pred_seq), self._frames(true_seq))

        # no gradient needed for the real branch
        with torch.no_grad():
            pred_real = self.D(true_f)
        pred_fake = self.D(pred_f)
        loss_adv = self._ragan(pred_fake, pred_real)
        parts['L_adv'] = loss_adv

        total = loss_pixel + self.beta_adv * loss_adv
        if self.use_perceptual:
            loss_percep = self._perceptual(pred_f, true_f)
            parts['L_percep'] = loss_percep
            total = total + loss_percep
        return total, parts

    def _discriminator_loss(self, pred_seq, true_seq):
        pred_f, true_f = self._subsample(self._frames(pred_seq).detach(),
                                         self._frames(true_seq))
        pred_real = self.D(true_f)
        pred_fake = self.D(pred_f)
        return self._ragan(pred_real, pred_fake)

    def generator_loss(self, pred_seq, true_seq):
        return self._retry_on_oom(self._generator_loss, pred_seq, true_seq)

    def discriminator_loss(self, pred_seq, true_seq):
        return self._retry_on_oom(self._discriminator_loss, pred_seq, true_seq)

    def forward(self, pred_seq, true_seq):
        return self.generator_loss(pred_seq, true_seq)
