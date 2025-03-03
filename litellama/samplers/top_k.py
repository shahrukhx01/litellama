import torch


class TopK:
    def sample(self, probs: torch.Tensor, k: int) -> torch.Tensor:
        """Samples from the probability distribution using the top-k approach.

        Args:
            probs (torch.Tensor): Probability distribution from which to sample.
            k (int): Number of top probabilities to consider.

        Returns:
            torch.Tensor: The sampled token index.
        """
        indices_to_remove = probs < torch.topk(probs, k)[0][..., -1, None]
        probs_filtered = probs.clone()
        probs_filtered[indices_to_remove] = 0

        # Redistribute probability mass to keep the sum to 1
        probs_filtered.div_(probs_filtered.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_filtered, num_samples=1)
        return next_token
