import torch
import torch.nn as nn
import torch.optim as optim


class SDFNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = 3
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gt_sdf_sphere(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Ground-truth SDF for a sphere centered at origin."""
    return torch.linalg.norm(points, dim=-1, keepdim=True) - radius


def sample_points(num_points: int, bound: float = 1.5, device: str = "cpu") -> torch.Tensor:
    return (torch.rand(num_points, 3, device=device) * 2.0 - 1.0) * bound


def train(
    epochs: int = 2000,
    batch_size: int = 4096,
    lr: float = 1e-3,
    radius: float = 1.0,
    device: str = "cpu",
) -> SDFNetwork:
    model = SDFNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        points = sample_points(batch_size, device=device)
        gt_sdf = gt_sdf_sphere(points, radius=radius)

        pred_sdf = model(points)
        loss = ((pred_sdf - gt_sdf) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    return model


def quick_test(model: SDFNetwork, device: str = "cpu") -> None:
    model.eval()
    test_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [0.4, 0.4, 0.4],
        ],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        pred = model(test_points).squeeze(-1)
        gt = gt_sdf_sphere(test_points).squeeze(-1)

    print("\nQuick test (pred vs gt):")
    for p, p_pred, p_gt in zip(test_points.cpu(), pred.cpu(), gt.cpu()):
        print(f"point={p.numpy()} pred={p_pred.item(): .5f} gt={p_gt.item(): .5f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train(epochs=1200, batch_size=4096, lr=1e-3, radius=1.0, device=device)
    quick_test(model, device=device)

    torch.save(model.state_dict(), "sdf_sphere_model.pt")
    print("\nSaved model to sdf_sphere_model.pt")
