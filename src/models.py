import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualWordCNN(nn.Module):
    """Deep CNN mapping spatial visual word grids to compact L2-normalized embeddings."""
    def __init__(self, in_channels=128, embedding_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (12, 12) -> (6, 6)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (Batch, 256, 1, 1)

            nn.Flatten(),  # -> (Batch, 256)
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


def train_triplet_cnn(model, data_loader, device, epochs=15, lr=1e-3, margin=0.2):
    """Trains VisualWordCNN using Triplet Margin Loss."""
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    print(f"[TRAIN] Beginning Triplet CNN Training ({epochs} Epochs)...")

    for epoch in range(epochs):
        running_loss = 0.0

        for anchor, positive, negative in data_loader:
            # Move batch to device
            a, p, n = (t.to(device, non_blocking=True) for t in (anchor, positive, negative))

            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {running_loss / len(data_loader):.5f}")

    return model

def build_spatial_pyramid(grids):
    """
    Zet 3D spatiële rasters van vorm (N, K, 12, 12) om naar een Spatial Pyramid.
    Niveaus:
      - Level 0: 1x1 (Globaal, gewicht 0.25) -> (N, K*1)
      - Level 1: 2x2 (Groef, gewicht 0.25)   -> (N, K*4)
      - Level 2: 4x4 (Fijn, gewicht 0.50)    -> (N, K*16)
    Totaal aantal dimensies: K * (1 + 4 + 16) = 21 * K (bij K=128 is dit 2.688).
    """
    N, K, H, W = grids.shape

    # 1x1 pooling --> sum over entire width and height
    L0 = grids.sum(axis=(2, 3)) * 0.25
    # 2x2 pooling --> split grid into blocks
    L1 = grids.reshape(N, K, 2, 6, 2, 6).sum(axis=(3, 5)).reshape(N, -1) * 0.25
    # 4x4 pooling
    L2 = grids.reshape(N, K, 4, 3, 4, 3).sum(axis=(3, 5)).reshape(N, -1) * 0.50

    # Stick all 3 levels behind each image
    return np.concatenate([L0, L1, L2], axis=1)