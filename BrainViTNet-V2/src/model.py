"""
BrainViTNet-V2 — Model Architecture
IEEE TMI / Medical Image Analysis Target
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import cv2

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# NOVEL COMPONENT ①: 2.5D SLICE FUSION MODULE
# ================================================================
class SliceFusion2_5D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_channels, out_channels), nn.Sigmoid())

    def forward(self, x):
        fused = self.fusion(torch.cat([
            self.branch_3x3(x), self.branch_5x5(x), self.branch_7x7(x)], dim=1))
        gate_weight = self.gate(fused).unsqueeze(-1).unsqueeze(-1)
        return x + gate_weight * fused


# ================================================================
# NOVEL COMPONENT ②: DOMAIN-ADAPTIVE BATCH NORMALIZATION
# ================================================================
class DomainAdaptiveBN(nn.Module):
    def __init__(self, num_features, num_domains=3, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        for i in range(num_domains):
            self.register_buffer(f'running_mean_{i}', torch.zeros(num_features))
            self.register_buffer(f'running_var_{i}', torch.ones(num_features))
        self.domain_gate = nn.Parameter(torch.ones(num_domains) / num_domains)
        self.current_domain = 0

    def set_domain(self, domain_id):
        self.current_domain = domain_id

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            running_mean = getattr(self, f'running_mean_{self.current_domain}')
            running_var = getattr(self, f'running_var_{self.current_domain}')
            running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            gate = F.softmax(self.domain_gate, dim=0)
            mean = sum(gate[i] * getattr(self, f'running_mean_{i}')
                       for i in range(self.num_domains))
            var = sum(gate[i] * getattr(self, f'running_var_{i}')
                      for i in range(self.num_domains))
        x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


# ================================================================
# ENHANCED CBAM WITH DEFORMABLE OFFSETS
# ================================================================
class EnhancedCBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)
        self.max_pool_c = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size,
                                      padding=kernel_size // 2, bias=False)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 3, padding=1, bias=False),
            nn.Tanh())
        self.offset_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        channel_att = torch.sigmoid(
            self.channel_mlp(self.avg_pool_c(x)) +
            self.channel_mlp(self.max_pool_c(x)))
        x_ca = channel_att * x
        offsets = self.offset_predictor(x_ca) * self.offset_scale
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device), indexing='ij')
        base_grid = (torch.stack([grid_x, grid_y], dim=-1)
                     .unsqueeze(0).expand(B, -1, -1, -1))
        offset_grid = (base_grid + offsets.permute(0, 2, 3, 1)).clamp(-1, 1)
        x_def = F.grid_sample(x_ca, offset_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        avg_s = torch.mean(x_def, dim=1, keepdim=True)
        max_s, _ = torch.max(x_def, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return spatial_att * x_def


# ================================================================
# NOVEL COMPONENT ③: DEFORMABLE CROSS-SCALE ATTENTION (DCSA)
# ================================================================
class DeformableCrossScaleAttention(nn.Module):
    def __init__(self, channels_list, out_channels=512):
        super().__init__()
        self.num_scales = len(channels_list)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for c in channels_list])
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels * self.num_scales, self.num_scales),
            nn.Softmax(dim=1))
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4), nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, feature_list, target_size):
        projected = [
            F.interpolate(self.projections[i](f), size=target_size,
                          mode='bilinear', align_corners=False)
            for i, f in enumerate(feature_list)]
        concat_pooled = torch.cat(
            [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in projected], dim=1)
        weights = self.scale_attention(concat_pooled)
        fused = sum(weights[:, i:i+1, None, None] * projected[i]
                    for i in range(self.num_scales))
        fused = fused * self.se_gate(fused).unsqueeze(-1).unsqueeze(-1)
        return self.refine(fused)


# ================================================================
# PATCH EMBEDDING
# ================================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


# ================================================================
# VISION TRANSFORMER BLOCK
# ================================================================
class ViTBlock(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, nhead,
                                          batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm)[0])
        return x + self.ff(self.norm2(x))


# ================================================================
# NOVEL COMPONENT ④: ALEATORIC UNCERTAINTY HEAD
# FIX: log_var clamped to [-4, 4] to prevent NaN
# ================================================================
class AleatoricUncertaintyHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.mean_head = nn.Linear(embed_dim, num_classes)
        self.log_var_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, num_classes))

    def forward(self, x):
        logits = self.mean_head(x)
        log_var = self.log_var_head(x).clamp(-4, 4)   # FIX: safe range
        return logits, log_var


# ================================================================
# NOVEL COMPONENT ⑤: CROSS-DOMAIN CONTRASTIVE LOSS
# FIX: stable logsumexp + safe division
# ================================================================
class CrossDomainContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        dev = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=dev)
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(dev)
        logits_mask = 1 - torch.eye(B, device=dev)
        mask = mask * logits_mask
        if mask.sum() == 0:
            return torch.tensor(0.0, device=dev)
        sim_matrix = sim_matrix - sim_matrix.detach().max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(
            exp_logits.sum(1, keepdim=True).clamp(min=1e-8))
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob.mean()


# ================================================================
# NOVEL COMPONENT ⑥: GRAD-CAM++
# ================================================================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        grads, acts = self.gradients, self.activations
        g2, g3 = grads**2, grads**3
        denom = 2*g2 + acts.sum((2, 3), keepdim=True)*g3 + 1e-7
        alpha = (g2 / denom) * torch.clamp(grads, min=0)
        weights = alpha.sum((2, 3), keepdim=True)
        heatmap = F.relu((weights * acts).sum(1).squeeze())
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.cpu().numpy()


def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    heatmap = plt.get_cmap(colormap_name)(activation)[:, :, :3]
    heatmap = cv2.resize(np.float32(heatmap),
                         (org_im.shape[1], org_im.shape[0]))
    cam = heatmap + np.float32(org_im)
    return np.uint8(255 * cam / np.max(cam))


# ================================================================
# BRAINVITNET-V2: COMPLETE ARCHITECTURE
# ================================================================
class BrainViTNetV2(nn.Module):
    def __init__(self, num_classes=4, embed_dim=512, patch_size=2,
                 nhead=8, ff_dim=1024, num_domains=3, mc_dropout_rate=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.slice_fusion = SliceFusion2_5D(3, 3)

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.cbam2 = EnhancedCBAM(512)
        self.cbam3 = EnhancedCBAM(1024)
        self.cbam4 = EnhancedCBAM(2048)

        self.dcsa = DeformableCrossScaleAttention([512, 1024, 2048], embed_dim)
        self.patch_embed = PatchEmbed(embed_dim, embed_dim, patch_size)
        self.vit_block1 = ViTBlock(embed_dim, nhead, ff_dim, mc_dropout_rate)
        self.vit_block2 = ViTBlock(embed_dim, nhead, ff_dim, mc_dropout_rate)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)

        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_dropout = nn.Dropout(0.3)
        self.aleatoric_head = AleatoricUncertaintyHead(embed_dim, num_classes)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 128))

    def _encode(self, x):
        x = self.slice_fusion(x)
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.cbam2(self.layer2(x))
        f3 = self.cbam3(self.layer3(f2))
        f4 = self.cbam4(self.layer4(f3))
        fused = self.dcsa([f2, f3, f4], (f4.shape[2], f4.shape[3]))
        tokens = self.patch_embed(fused)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]
        tokens = self.vit_block2(self.vit_block1(tokens))
        tokens = self.mc_dropout(tokens)
        return self.pool(tokens.transpose(1, 2)).squeeze(-1)

    def forward(self, x, return_features=False):
        pooled = self.classifier_dropout(self._encode(x))
        logits, log_var = self.aleatoric_head(pooled)
        if return_features:
            return logits, log_var, self.contrastive_projector(pooled.detach())
        return logits, log_var

    def predict_with_uncertainty(self, x, T=10):
        self.train()
        logits_list, logvar_list = [], []
        with torch.no_grad():
            for _ in range(T):
                lg, lv = self(x)
                logits_list.append(torch.softmax(lg, dim=1))
                logvar_list.append(lv)
        logits_stack = torch.stack(logits_list)
        logvar_stack = torch.stack(logvar_list)
        self.eval()
        return (logits_stack.mean(0),
                logits_stack.var(0).sum(1),
                torch.exp(logvar_stack).mean(0).sum(1))


# ================================================================
# UNCERTAINTY-AWARE LOSS
# FIX: clamped precision, stable formulation
# ================================================================
class UncertaintyAwareLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, log_var, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_weight = (1 - torch.exp(-ce_loss).detach()) ** self.gamma
        lv = log_var.gather(1, targets.unsqueeze(1)).squeeze(1)
        precision = torch.exp(-lv).clamp(max=10.0)
        unc_loss = 0.5 * precision * focal_weight * ce_loss + 0.5 * lv
        return unc_loss.mean()


# ================================================================
# TEMPERATURE SCALING
# ================================================================
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.1)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
