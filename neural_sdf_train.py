import argparse
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


def gt_sdf_box(points: torch.Tensor, half_size: torch.Tensor) -> torch.Tensor:
    """Ground-truth SDF for an axis-aligned box centered at origin."""
    q = torch.abs(points) - half_size
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1, keepdim=True)
    inside = torch.clamp(torch.max(q, dim=-1, keepdim=True).values, max=0.0)
    return outside + inside


def translate(points: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return points - offset


def sdf_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def gt_sdf_composite(points: torch.Tensor) -> torch.Tensor:
    """Composite shape: union of sphere and box, then carve with another sphere."""
    sphere_main = gt_sdf_sphere(translate(points, torch.tensor([-0.25, 0.0, 0.0], device=points.device)), radius=0.7)
    box_main = gt_sdf_box(
        translate(points, torch.tensor([0.3, 0.0, 0.0], device=points.device)),
        half_size=torch.tensor([0.45, 0.35, 0.35], device=points.device),
    )
    base = sdf_union(sphere_main, box_main)
    hole = gt_sdf_sphere(translate(points, torch.tensor([0.15, 0.0, 0.0], device=points.device)), radius=0.22)
    return torch.maximum(base, -hole)


def get_gt_sdf(points: torch.Tensor, shape_type: str, radius: float = 1.0) -> torch.Tensor:
    if shape_type == "sphere":
        return gt_sdf_sphere(points, radius=radius)
    if shape_type == "box":
        return gt_sdf_box(points, half_size=torch.tensor([0.7, 0.5, 0.45], device=points.device))
    if shape_type == "composite":
        return gt_sdf_composite(points)
    raise ValueError(f"Unsupported shape_type: {shape_type}")


def sample_points(num_points: int, bound: float = 1.5, device: str = "cpu") -> torch.Tensor:
    return (torch.rand(num_points, 3, device=device) * 2.0 - 1.0) * bound


def train(
    epochs: int = 2000,
    batch_size: int = 4096,
    lr: float = 1e-3,
    radius: float = 1.0,
    shape_type: str = "sphere",
    eikonal_weight: float = 0.1,
    device: str = "cpu",
) -> SDFNetwork:
    model = SDFNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        points = sample_points(batch_size, device=device)
        points.requires_grad_(True)
        gt_sdf = get_gt_sdf(points, shape_type=shape_type, radius=radius)

        pred_sdf = model(points)
        sdf_loss = ((pred_sdf - gt_sdf) ** 2).mean()
        eik_loss = eikonal_loss(pred_sdf, points)
        loss = sdf_loss + eikonal_weight * eik_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | Shape: {shape_type:<9} | Total: {loss.item():.6f} "
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
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def extract_mesh_marching_cubes(
    model: SDFNetwork,
    resolution: int = 96,
    bound: float = 1.3,
    level: float = 0.0,
    device: str = "cpu",
    out_path: str = "mesh.obj",
) -> None:
    sdf_grid = evaluate_grid_sdf(model, resolution=resolution, bound=bound, device=device)
    sdf_np = sdf_grid.detach().cpu().numpy()

    min_sdf = float(sdf_np.min())
    max_sdf = float(sdf_np.max())
    iso_level = level
    if not (min_sdf <= level <= max_sdf):
        iso_level = min(max(level, min_sdf), max_sdf)
        print(
            f"Requested level={level:.4f} is outside SDF range [{min_sdf:.4f}, {max_sdf:.4f}]. "
            f"Using clamped level={iso_level:.4f}."
        )

    spacing = (2.0 * bound) / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(
        sdf_np,
        level=iso_level,
        spacing=(spacing, spacing, spacing),
    )
    verts -= bound

    save_obj(verts, faces, out_path)
    print(f"Saved mesh to {out_path} (verts={len(verts)}, faces={len(faces)})")


def quick_test(model: SDFNetwork, shape_type: str = "sphere", radius: float = 1.0, device: str = "cpu") -> None:
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
        gt = get_gt_sdf(test_points, shape_type=shape_type, radius=radius).squeeze(-1)

    print("\nQuick test (pred vs gt):")
    for p, p_pred, p_gt in zip(test_points.cpu(), pred.cpu(), gt.cpu()):
        print(f"point={p.numpy()} pred={p_pred.item(): .5f} gt={p_gt.item(): .5f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neural SDF and export mesh with Marching Cubes.")
    parser.add_argument("--shape", choices=["sphere", "box", "composite"], default="sphere")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--eikonal-weight", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--bound", type=float, default=1.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    output_dir = Path("generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"sdf_{args.shape}_model.pt"
    mesh_path = output_dir / f"mesh_{args.shape}.obj"

    model = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        radius=args.radius,
        shape_type=args.shape,
        eikonal_weight=args.eikonal_weight,
        device=device,
    )
    quick_test(model, shape_type=args.shape, radius=args.radius, device=device)

    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    extract_mesh_marching_cubes(
        model,
        resolution=args.resolution,
        bound=args.bound,
        level=0.0,
        device=device,
        out_path=str(mesh_path),
    )
