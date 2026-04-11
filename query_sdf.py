import argparse
from pathlib import Path

import torch

from neural_sdf_train import SDFNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query SDF values from a saved Neural SDF model (.pt).")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument(
        "--point",
        dest="points",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="3D point to query (can be specified multiple times)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Inference device",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return device_arg


def build_model(model_path: Path, device: str) -> SDFNetwork:
    model = SDFNetwork().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def query_points(model: SDFNetwork, points: list[list[float]], device: str) -> torch.Tensor:
    x = torch.tensor(points, dtype=torch.float32, device=device)
    with torch.no_grad():
        sdf = model(x).squeeze(-1)
    return sdf.cpu()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = resolve_device(args.device)
    model = build_model(model_path, device)

    points = args.points if args.points else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.3, 0.0]]
    sdf_values = query_points(model, points, device)

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("Query results:")
    for p, sdf in zip(points, sdf_values.tolist()):
        print(f"point={p} sdf={sdf:.6f}")


if __name__ == "__main__":
    main()
