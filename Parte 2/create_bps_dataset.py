import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np
#pip install bps
from bps import bps
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


@dataclass
class ProcessedSample:
    """Container for a processed point cloud sample."""
    bps_features: np.ndarray
    label: int
    points_norm: np.ndarray
    centroid: np.ndarray
    scale: float


# =========================================================
# OFF loader
# =========================================================
def load_off(filepath: str) -> Optional[np.ndarray]:
    """Load vertices from an OFF file with improved error handling."""
    try:
        with open(filepath, "r") as f:
            # Read all non-empty lines at once
            lines = [l.strip() for l in f if l.strip()]
        
        if not lines:
            return None
        
        # Handle OFF header
        if lines[0] == "OFF":
            header_idx = 1
        elif lines[0].startswith("OFF"):
            lines[0] = lines[0][3:]  # Strip "OFF" prefix
            header_idx = 0
        else:
            return None
        
        # Parse header
        n_verts = int(lines[header_idx].split()[0])
        
        # Pre-allocate array for better performance
        verts = np.empty((n_verts, 3), dtype=np.float32)
        
        # Parse vertices using numpy (faster than list append)
        for i, line in enumerate(lines[header_idx + 1 : header_idx + 1 + n_verts]):
            verts[i] = list(map(float, line.split()[:3]))
        
        # Validation
        if len(verts) < 10 or not np.isfinite(verts).all():
            return None
        
        return verts
    
    except (ValueError, IndexError, IOError):
        return None


def sample_points(verts: np.ndarray, n: int) -> np.ndarray:
    """Sample n points from vertices with replacement if needed."""
    idx = np.random.choice(len(verts), size=n, replace=(len(verts) < n))
    return verts[idx]


# =========================================================
# Normalization
# =========================================================
def normalize_pc(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center and scale point cloud to unit sphere.
    
    Returns:
        Tuple of (normalized_points, centroid, scale)
    """
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    scale = np.linalg.norm(pts_centered, axis=1).max() + 1e-8
    pts_normalized = pts_centered / scale
    
    return pts_normalized.astype(np.float32), centroid.astype(np.float32), float(scale)


# =========================================================
# File collection
# =========================================================
def collect_files(data_root: str) -> Tuple[Dict[str, List], List[str]]:
    """Collect all OFF files organized by train/test splits."""
    data_path = Path(data_root)
    
    # Get sorted class directories
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class2idx = {c: i for i, c in enumerate(classes)}
    splits = {"train": [], "test": []}
    
    for cls in classes:
        for split in ("train", "test"):
            split_dir = data_path / cls / split
            if not split_dir.is_dir():
                continue
            
            # Use pathlib's glob for cleaner file collection
            for fpath in sorted(split_dir.glob("*.off")):
                splits[split].append((str(fpath), class2idx[cls]))
    
    return splits, classes


# =========================================================
# Sample processing
# =========================================================
def process_single_sample(
    file_info: Tuple[str, int],
    basis_points: np.ndarray,
    num_points: int,
    seed_offset: int
) -> Optional[ProcessedSample]:
    """Process a single OFF file into BPS features.
    
    This function is designed to be called in parallel.
    """
    fpath, label = file_info
    
    # Set seed for reproducibility (unique per sample)
    np.random.seed(seed_offset + hash(fpath) % 100000)
    
    # Load and validate
    verts = load_off(fpath)
    if verts is None:
        return None
    
    # Sample and normalize
    pts = sample_points(verts, num_points)
    pts_norm, centroid, scale = normalize_pc(pts)
    
    # BPS encoding
    try:
        feat = bps.encode(
            pts_norm[None],
            bps_arrangement='custom',
            bps_cell_type='deltas',
            custom_basis=basis_points,
            verbose=0,
            n_jobs=1,
        )[0]  # (n_bps, 3)
    except Exception:
        return None
    
    return ProcessedSample(
        bps_features=feat,
        label=label,
        points_norm=pts_norm,
        centroid=centroid,
        scale=scale
    )


# =========================================================
# Parallel split preparation
# =========================================================
def prepare_split_parallel(
    file_list: List[Tuple[str, int]],
    basis_points: np.ndarray,
    num_points: int,
    tag: str,
    seed: int,
    n_workers: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a complete split using parallel processing."""
    
    n_workers = n_workers or min(os.cpu_count() or 1, len(file_list))
    
    # Prepare processing function with fixed parameters
    process_fn = partial(
        process_single_sample,
        basis_points=basis_points,
        num_points=num_points,
        seed_offset=seed
    )
    
    results = []
    skipped = 0
    
    print(f"  [{tag}] Processing with {n_workers} workers...")
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_fn, file_info): i 
                   for i, file_info in enumerate(file_list)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is None:
                    print(f"  SKIP {file_list[idx][0]}")
                    skipped += 1
                else:
                    results.append(result)
                
                if (idx + 1) % 100 == 0:
                    print(f"  [{tag}] {idx + 1}/{len(file_list)} submitted...")
            except Exception as e:
                print(f"  ERROR processing {file_list[idx][0]}: {e}")
                skipped += 1
    
    print(f"  [{tag}] Done. Processed: {len(results)}, Skipped: {skipped}/{len(file_list)}")
    
    # Consolidate results
    if not results:
        raise ValueError(f"No valid samples processed for {tag} split!")
    
    bps_arr = np.stack([r.bps_features for r in results], axis=0)
    labels = np.array([r.label for r in results], dtype=np.int64)
    points = np.stack([r.points_norm for r in results], axis=0)
    centroids = np.stack([r.centroid for r in results], axis=0)
    scales = np.array([r.scale for r in results], dtype=np.float32)
    
    return bps_arr, labels, points, centroids, scales


# =========================================================
# Standardization and saving
# =========================================================
def compute_standardization_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for feature standardization."""
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    return feat_mean, feat_std


def save_split_data(
    out_dir: str,
    split: str,
    bps_std: np.ndarray,
    labels: np.ndarray,
    points: np.ndarray,
    centroids: np.ndarray,
    scales: np.ndarray
) -> None:
    """Save all arrays for a given split."""
    np.save(os.path.join(out_dir, f"{split}_bps_std.npy"), bps_std)
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), labels)
    np.save(os.path.join(out_dir, f"{split}_points_norm.npy"), points)
    np.save(os.path.join(out_dir, f"{split}_centroids.npy"), centroids)
    np.save(os.path.join(out_dir, f"{split}_scales.npy"), scales)


# =========================================================
# Main
# =========================================================
def main(args):
    """Main dataset preparation pipeline."""
    # Setup
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Collect files
    splits, classes = collect_files(args.data_root)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train files: {len(splits['train'])}  |  Test files: {len(splits['test'])}")
    
    # Generate basis points
    np.random.seed(args.seed)
    basis = bps.generate_random_basis(
        n_points=args.n_bps_points,
        n_dims=3,
        radius=args.bps_radius,
        random_seed=args.seed,
    )
    np.save(os.path.join(args.out_dir, "basis_points.npy"), basis)
    print(f"Basis points: {basis.shape}, radius={args.bps_radius}")
    
    # -- Train --
    print("\n" + "="*60)
    print("[TRAIN SPLIT]")
    print("="*60)
    tr_bps, tr_labels, tr_pts, tr_c, tr_s = prepare_split_parallel(
        splits["train"], basis, args.num_points, "train", args.seed, args.n_workers
    )
    
    # Compute and apply standardization
    feat_mean, feat_std = compute_standardization_stats(tr_bps)
    tr_bps_std = (tr_bps - feat_mean) / feat_std
    
    # Save train data
    save_split_data(args.out_dir, "train", tr_bps_std, tr_labels, tr_pts, tr_c, tr_s)
    np.save(os.path.join(args.out_dir, "feature_mean.npy"), feat_mean)
    np.save(os.path.join(args.out_dir, "feature_std.npy"), feat_std)
    
    print(f"  train_bps_std: {tr_bps_std.shape}  "
          f"mean={tr_bps_std.mean():.4f}  std={tr_bps_std.std():.4f}")
    
    # -- Test --
    print("\n" + "="*60)
    print("[TEST SPLIT]")
    print("="*60)
    te_bps, te_labels, te_pts, te_c, te_s = prepare_split_parallel(
        splits["test"], basis, args.num_points, "test", args.seed, args.n_workers
    )
    
    # Apply train standardization to test
    te_bps_std = (te_bps - feat_mean) / feat_std
    
    # Save test data
    save_split_data(args.out_dir, "test", te_bps_std, te_labels, te_pts, te_c, te_s)
    
    print(f"  test_bps_std: {te_bps_std.shape}  "
          f"mean={te_bps_std.mean():.4f}  std={te_bps_std.std():.4f}")
    
    # -- Metadata --
    meta = {
        "seed": args.seed,
        "n_bps_points": args.n_bps_points,
        "bps_radius": args.bps_radius,
        "num_points": args.num_points,
        "classes": classes,
        "n_train": int(len(tr_labels)),
        "n_test": int(len(te_labels)),
        "n_workers": args.n_workers or os.cpu_count(),
    }
    
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(args.out_dir, "classes.npy"), np.array(classes))
    
    print("\n" + "="*60)
    print(f"✓ Dataset saved to: {args.out_dir}")
    print(f"✓ Train samples: {len(tr_labels)}")
    print(f"✓ Test samples: {len(te_labels)}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare BPS-encoded point cloud dataset from ModelNet OFF files"
    )
    parser.add_argument("--data_root", type=str, default="ModelNet3",
                        help="Root directory containing class folders")
    parser.add_argument("--out_dir", type=str, default="bps_dataset",
                        help="Output directory for processed dataset")
    
    
    parser.add_argument("--num_points", type=int, default=1000,
                        help="Number of points to sample from each mesh")
    # Probar con 512, 1024
    parser.add_argument("--n_bps_points", type=int, default=512,
                        help="Number of basis points for BPS encoding")
    parser.add_argument("--bps_radius", type=float, default=1.1,
                        help="Radius for BPS basis point generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    main(args)