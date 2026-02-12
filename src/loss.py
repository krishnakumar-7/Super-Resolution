import torch
import torch.nn as nn
import numpy as np


class PhysicsLoss(nn.Module):

    def __init__(self, target_spectrum_file, device='cuda', size=1024):
        super().__init__()
        self.device = device

        target_np = np.load(target_spectrum_file)
        self.target_spectrum = torch.from_numpy(target_np).float().to(device)
        self.max_k = int(self.target_spectrum.shape[0])
        self.criterion = nn.MSELoss()

        # Precompute radial k-magnitude grid
        k_freq = torch.fft.fftfreq(size).to(device) * size
        kx, ky = torch.meshgrid(k_freq, k_freq, indexing='ij')
        self.k_mag = torch.sqrt(kx**2 + ky**2).round().long().flatten()

    def get_spectrum(self, hr_field):
        b, c, h, w = hr_field.shape
        fft_im = torch.fft.fft2(hr_field)

        # Compute power spectrum manually to avoid complex tensor issues
        power_spectrum = fft_im.real**2 + fft_im.imag**2
        energy_2d = torch.sum(power_spectrum, dim=1).view(b, -1)

        # Binning
        spectrum = torch.zeros((b, self.max_k + 1), device=hr_field.device)
        indices = self.k_mag.expand(b, -1)
        mask = indices < self.max_k

        spectrum.scatter_add_(1, indices * mask, energy_2d * mask)

        # Log spectrum, excluding DC component
        return torch.log(spectrum[:, 1:self.max_k+1] + 1e-8)

    def forward(self, hr_guess):
        pred_log_spec = self.get_spectrum(hr_guess)

        target_log_spec = torch.log(self.target_spectrum + 1e-8)
        target_log_spec = target_log_spec.unsqueeze(
            0).expand(pred_log_spec.shape[0], -1)

        n_bins = min(pred_log_spec.shape[1], target_log_spec.shape[1])
        return self.criterion(pred_log_spec[:, :n_bins], target_log_spec[:, :n_bins])


class DivergenceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        # Central finite difference
        du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
        dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0

        div = du_dx[:, 1:-1, :] + dv_dy[:, :, 1:-1]
        return torch.mean(div**2)
