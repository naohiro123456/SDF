import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from skimage import measure


def eikonal_loss(pred_sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Regularize SDF gradients so |grad f| ~= 1."""
    gradients = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(dim=-1) - 1.0) ** 2).mean()


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
    eikonal_weight: float = 0.1,
    device: str = "cpu",
) -> SDFNetwork:
    model = SDFNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        points = sample_points(batch_size, device=device)
        points.requires_grad_(True)
        gt_sdf = gt_sdf_sphere(points, radius=radius)

        pred_sdf = model(points)
        sdf_loss = ((pred_sdf - gt_sdf) ** 2).mean()
        eik_loss = eikonal_loss(pred_sdf, points)
        loss = sdf_loss + eikonal_weight * eik_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | Total: {loss.item():.6f} "
                f"| SDF: {sdf_loss.item():.6f} | Eik: {eik_loss.item():.6f}"
            )

    return model


def evaluate_grid_sdf(
    model: SDFNetwork,
    resolution: int = 96,
    bound: float = 1.3,
    device: str = "cpu",
    chunk_size: int = 65536,
) -> torch.Tensor:
    coords = torch.linspace(-bound, bound, resolution, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    sdf_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, grid_points.shape[0], chunk_size):
            end = start + chunk_size
            chunk = grid_points[start:end]
            sdf_values.append(model(chunk).squeeze(-1))

    return torch.cat(sdf_values, dim=0).reshape(resolution, resolution, resolution)


def save_obj(vertices, faces, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed.
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def extract_mesh_marching_cubes(
    model: SDFNetwork,
    resolution: int = 96,
    bound: float = 1.3,
    level: float = 0.0,
    device: str = "cpu",
    out_path: str = "neural_sdf_mesh.obj",
) -> None:
    sdf_grid = evaluate_grid_sdf(model, resolution=resolution, bound=bound, device=device)
    sdf_np = sdf_grid.detach().cpu().numpy()

    spacing = (2.0 * bound) / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(sdf_np, level=level, spacing=(spacing, spacing, spacing))
    verts -= bound

    save_obj(verts, faces, out_path)
    print(f"Saved mesh to {out_path} (verts={len(verts)}, faces={len(faces)})")


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
    output_dir = Path("generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "sdf_sphere_model.pt"
    mesh_path = output_dir / "neural_sdf_mesh.obj"

    model = train(
        epochs=1200,
        batch_size=4096,
        lr=1e-3,
        radius=1.0,
        eikonal_weight=0.1,
        device=device,
    )
    quick_test(model, device=device)

    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    extract_mesh_marching_cubes(
        model,
        resolution=96,
        bound=1.3,
        level=0.0,
        device=device,
        out_path=str(mesh_path),
    )
