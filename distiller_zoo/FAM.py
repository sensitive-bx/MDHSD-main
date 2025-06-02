import torch
import torch.nn as nn
import torch.nn.functional as F


class FAMLoss(nn.Module):
    """
    Frequency Attention for Knowledge Distillation

    This implements the FAM (Frequency Attention Module) distillation loss from the paper
    "Frequency Attention for Knowledge Distillation"
    """

    def __init__(self, student_channels, teacher_channels, alpha=0.5):
        super(FAMLoss, self).__init__()

        self.alpha = alpha  # Balancing parameter between spatial and frequency domain

        # Adapter layers if student and teacher channels differ
        self.adaptation = nn.ModuleList([])
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            self.adaptation.append(
                nn.Sequential(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(t_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # Frequency Attention Module components
        self.frequency_branch = nn.ModuleList([])
        self.spatial_branch = nn.ModuleList([])
        self.combine_weights = nn.ParameterList([])

        for ch in teacher_channels:
            # Frequency branch - processes features in frequency domain
            self.frequency_branch.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True)
                )
            )

            # Spatial branch - processes features in spatial domain
            self.spatial_branch.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True)
                )
            )

            # Learnable weight parameter to balance frequency and spatial branches
            self.combine_weights.append(nn.Parameter(torch.tensor(0.5)))

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: list of student's intermediate features
            teacher_features: list of teacher's intermediate features

        Returns:
            loss: frequency attention distillation loss
        """
        loss = 0
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # Skip if dimensions don't match
            if len(self.adaptation) <= i:
                continue

            # Adapt student features to match teacher dimensions
            s_feat = self.adaptation[i](s_feat)

            # Resize student features to match teacher spatial dimensions if needed
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)

            # Apply Frequency Attention Module
            s_feat_enhanced = self.apply_fam(s_feat, i)

            # Calculate L2 loss between enhanced student features and teacher features
            loss += F.mse_loss(s_feat_enhanced, t_feat)

        return loss

    def apply_fam(self, x, idx):
        """
        Apply Frequency Attention Module to feature maps

        Args:
            x: input feature map
            idx: index to select corresponding module

        Returns:
            y: enhanced feature map
        """
        B, C, H, W = x.shape

        # 1. Convert to frequency domain using FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))

        # 2. Process in frequency domain - magnitude only approach
        x_freq_magnitude = torch.abs(x_freq)
        x_freq_phase = torch.angle(x_freq)

        # 3. Apply frequency domain processing
        freq_processed = self.frequency_branch[idx](x_freq_magnitude)

        # 4. Reconstruct the frequency domain representation
        x_freq_new = freq_processed * torch.exp(1j * x_freq_phase)

        # 5. Convert back to spatial domain using IFFT
        x_freq_enhanced = torch.fft.irfft2(x_freq_new, s=(H, W), dim=(-2, -1))

        # 6. Apply spatial processing branch
        x_spatial = self.spatial_branch[idx](x)

        # 7. Combine frequency and spatial branches with learnable weight
        weight = torch.sigmoid(self.combine_weights[idx])
        enhanced = weight * x_freq_enhanced + (1 - weight) * x_spatial

        return enhanced

    def high_pass_filter(self, x_freq, r=0.5):
        """
        Apply high-pass filter in frequency domain

        Args:
            x_freq: frequency domain representation
            r: radius ratio for high-pass filter (0-1)

        Returns:
            x_freq_filtered: filtered frequency representation
        """
        B, C, H, W_half = x_freq.shape
        mask = torch.ones((H, W_half), device=x_freq.device)

        # Create distance matrix (from center)
        center_h, center_w = H // 2, 0
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=x_freq.device),
                                        torch.arange(W_half, device=x_freq.device))

        # Calculate distance from center
        distance = torch.sqrt((y_grid - center_h) ** 2 + (x_grid - center_w) ** 2)
        max_distance = torch.sqrt(torch.tensor(H ** 2 + W_half ** 2, device=x_freq.device))

        # High-pass filter: suppress low frequencies (center), keep high frequencies
        mask[distance < r * max_distance] = 0.1  # Attenuate low frequencies

        # Apply mask to frequency domain
        return x_freq * mask.unsqueeze(0).unsqueeze(0)