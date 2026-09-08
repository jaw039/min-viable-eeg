import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    Compact EEGNet-style model for EEG classification.

    Expected input shape:
        (batch, channels, samples)

    Example:
        (32, 64, 641)

    Output shape:
        (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int = 641,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # EEGNet commonly uses a 4-D input:
        # (batch, 1, channels, samples)
        #
        # We will add that extra dimension automatically
        # inside forward().

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(n_channels, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=16,
                bias=False,
            ),

            # Pointwise convolution
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                bias=False,
            ),

            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(dropout),
        )

        # Determine flattened feature size automatically.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                n_channels,
                n_samples,
            )

            features = self._forward_features(dummy)

            flattened_size = features.numel()

        self.classifier = nn.Linear(
            flattened_size,
            n_classes,
        )

    def _forward_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:

        # Convert:
        # (batch, channels, samples)
        #
        # into:
        # (batch, 1, channels, samples)
        X = X.unsqueeze(1)

        X = self.temporal_conv(X)

        X = self.depthwise_conv(X)

        X = self.separable_conv(X)

        return X

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:

        if X.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, channels, samples), "
                f"got {tuple(X.shape)}"
            )

        if X.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got {X.shape[1]}"
            )

        X = self._forward_features(X)

        X = torch.flatten(
            X,
            start_dim=1,
        )

        logits = self.classifier(X)

        return logits