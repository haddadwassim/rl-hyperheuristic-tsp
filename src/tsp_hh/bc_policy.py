import torch
import torch.nn as nn


class NextCityPolicy(nn.Module):
    """
    Simple behavior-cloning policy for constructive TSP.

    Input per candidate city:
      - candidate x, y
      - current city x, y
      - start city x, y
      - distance from current to candidate
      - distance from candidate to start
      - visited flag

    The model scores each candidate city independently.
    Invalid/visited cities are masked during action selection.
    """

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 128):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        candidate_features:
            shape = (batch_size, n_cities, feature_dim)

        returns:
            logits of shape (batch_size, n_cities)
        """
        scores = self.scorer(candidate_features).squeeze(-1)
        return scores