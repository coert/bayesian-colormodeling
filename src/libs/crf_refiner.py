import numpy as np
import torch

EPS = np.finfo(float).eps


class CRFRefiner(torch.nn.Module):
    def __init__(self, pairwise_weight=0.5, num_iterations=5, device="cpu"):
        """
        Simple CRF-like refiner for 2D image grids.
        :param pairwise_weight: weight for spatial smoothness term.
        :param num_iterations: number of refinement iterations.
        :param device: 'cpu' or 'cuda'.
        """
        super().__init__()
        self.pairwise_weight = pairwise_weight
        self.num_iterations = num_iterations
        self.device = device

    def forward(self, unary_probs):
        """
        :param unary_probs: torch.Tensor, shape [1, 1, H, W], values in (0, 1)
        :return: refined_probs: torch.Tensor, same shape
        """
        unary_probs = unary_probs.to(self.device)
        refined = unary_probs.clone()

        for _ in range(self.num_iterations):
            # Unary term (negative log likelihood)
            unary_energy = -torch.log(refined + EPS)

            # Pairwise term (smoothness: absolute difference with neighbors)
            pairwise_energy = torch.zeros_like(unary_energy)

            pairwise_energy[:, :, :-1, :] += (
                refined[:, :, :-1, :] - refined[:, :, 1:, :]
            ).abs()
            pairwise_energy[:, :, 1:, :] += (
                refined[:, :, 1:, :] - refined[:, :, :-1, :]
            ).abs()
            pairwise_energy[:, :, :, :-1] += (
                refined[:, :, :, :-1] - refined[:, :, :, 1:]
            ).abs()
            pairwise_energy[:, :, :, 1:] += (
                refined[:, :, :, 1:] - refined[:, :, :, :-1]
            ).abs()

            # Total energy
            total_energy = unary_energy + self.pairwise_weight * pairwise_energy

            # Update refined probabilities
            refined = torch.sigmoid(-total_energy)

            # Clamp to keep stability
            refined = torch.clamp(refined, 1e-5, 1 - 1e-5)

        # Normalize final result
        refined = refined - refined.min()
        refined = refined / (refined.max() + EPS)

        return refined
