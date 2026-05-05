import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_gray(path: str) -> np.ndarray:
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to load images")
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)

def compute_disparity(Il: np.ndarray,Ir: np.ndarray,window_size: int,cost: str,max_disp: int = 64) -> np.ndarray:
    assert Il.shape == Ir.shape, "Left and right images must be the same size."
    assert window_size % 2 == 1, "window_size must be odd."
    assert cost in ("SAD", "SSD"), "cost must be 'SAD' or 'SSD'."

    H, W = Il.shape
    half = window_size // 2
    disp_map = np.zeros((H, W), dtype=np.int32)
    pad = half
    Il_pad = np.pad(Il, pad, mode="edge")
    Ir_pad = np.pad(Ir, pad, mode="edge")
    for row in range(H):
        pr = row + pad
        left_strip = Il_pad[pr - half: pr + half + 1, :]   # (w, W+2p)

        for col in range(W):
            pc = col + pad
            left_patch = left_strip[:, pc - half: pc + half + 1]
            best_cost = np.inf
            best_d    = 0
            for d in range(0, max_disp + 1):
                rc = pc - d          # shifted column in padded right image
                if rc - half < 0:   # out of bounds → stop searching
                    break

                right_patch = Ir_pad[pr - half: pr + half + 1,rc - half: rc + half + 1]
                diff = left_patch - right_patch
                if cost == "SAD":
                    c_val = np.sum(np.abs(diff))
                else:               # SSD
                    c_val = np.sum(diff * diff)

                if c_val < best_cost:
                    best_cost = c_val
                    best_d    = d

            disp_map[row, col] = best_d

    return disp_map

def compute_disparity_fast(Il: np.ndarray,
                           Ir: np.ndarray,
                           window_size: int,
                           cost: str,
                           max_disp: int = 64) -> np.ndarray:
    from scipy.ndimage import uniform_filter  # lightweight SciPy dependency
    H, W = Il.shape
    half  = window_size // 2
    disp_map  = np.zeros((H, W), dtype=np.int32)
    best_cost = np.full((H, W), np.inf)
    for d in range(0, max_disp + 1):
        if d == 0:
            Ir_shifted = Ir
        else:
            Ir_shifted = np.zeros_like(Ir)
            Ir_shifted[:, d:] = Ir[:, :-d]   # columns [0..d-1] stay 0
        diff = Il - Ir_shifted
        if cost == "SAD":
            pixel_cost = np.abs(diff)
        else:
            pixel_cost = diff * diff
        block_cost = uniform_filter(pixel_cost, size=window_size) * (window_size ** 2)# Sum over w×w neighbourhood using a box filter
        mask = block_cost < best_cost
        best_cost[mask] = block_cost[mask]#keep track of the minimum cost disparity
        disp_map[mask]  = d
    return disp_map


def run(Il: np.ndarray,Ir: np.ndarray,max_disp: int = 64,out_dir: str = "."):
    window_sizes = [1, 5, 9]
    cost_fns     = ["SAD", "SSD"]
    results = {}

    print(f"\n{'='*62}")
    print(f"  Stereo Block Matching  |  image {Il.shape[1]}×{Il.shape[0]}")
    print(f"  max_disp={max_disp}   window sizes={window_sizes}")
    print(f"  {'-'*42}")

    for w in window_sizes:
        for cost in cost_fns:
            disp = compute_disparity_fast(Il, Ir, w, cost, max_disp)
            results[(w, cost)] = disp
            if HAS_PIL:
                d_norm = disp.astype(np.float32)
                if d_norm.max() > 0:
                    d_norm = (d_norm / d_norm.max() * 255).astype(np.uint8)
                fname = Path(out_dir) / f"disparity_w{w}_{cost}.png"
                Image.fromarray(d_norm).save(fname)
                print(f"Saved: {fname}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo block matching (SAD & SSD)")
    parser.add_argument("--max_disp", type=int, default=32, help="Maximum disparity")
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()
    image_pairs = [
        ("l1.png", "r1.png"),
        ("l2.png", "r2.png"),
        ("l3.png", "r3.png")
    ]
    for left_name, right_name in image_pairs:
        left_path = Path("stereo_materials") / left_name
        right_path = Path("stereo_materials") / right_name
        print(f"\nProcessing Pair: {left_name} vs {right_name}")
        if left_path.exists() and right_path.exists():
            Il = load_gray(str(left_path))
            Ir = load_gray(str(right_path))
            pair_out = Path(args.out) / left_name.split('.')[0]
            pair_out.mkdir(parents=True, exist_ok=True)
            run(Il, Ir, max_disp=args.max_disp, out_dir=str(pair_out))
        else:
            print(f"Skipping: Files {left_name} or {right_name} not found.")
