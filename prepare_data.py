import os
import argparse
import random
import shutil
from pathlib import Path

def prepare_data(real_dir, anime_dir, target_root, total_size, ratios):
    """
    Creates symlinks of images from NAS to a local data directory with train/val/test split.
    """
    # Parse ratios
    r_train, r_val, r_test = map(float, ratios.split(':'))
    total_ratio = r_train + r_val + r_test
    r_train /= total_ratio
    r_val /= total_ratio
    r_test /= total_ratio

    def collect_images(src_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        paths = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    paths.append(os.path.abspath(os.path.join(root, f)))
        return sorted(paths)

    print(f"Scanning {real_dir}...")
    real_paths = collect_images(real_dir)
    print(f"Scanning {anime_dir}...")
    anime_paths = collect_images(anime_dir)

    print(f"Found {len(real_paths)} real images and {len(anime_paths)} anime images.")

    # Limit to total_size
    if total_size > 0:
        if len(real_paths) > total_size:
            random.shuffle(real_paths)
            real_paths = real_paths[:total_size]
        if len(anime_paths) > total_size:
            random.shuffle(anime_paths)
            anime_paths = anime_paths[:total_size]

    # Ensure consistent count if needed, but for unpaired it's okay to be slightly different.
    # However, for training stability, let's keep them somewhat balanced.
    min_count = min(len(real_paths), len(anime_paths))
    print(f"Using {min_count} images from each domain.")
    
    random.shuffle(real_paths)
    random.shuffle(anime_paths)
    
    real_paths = real_paths[:min_count]
    anime_paths = anime_paths[:min_count]

    # Split
    train_idx = int(min_count * r_train)
    val_idx = train_idx + int(min_count * r_val)

    splits = {
        'train': (0, train_idx),
        'val': (train_idx, val_idx),
        'test': (val_idx, min_count)
    }

    # Create target dirs
    target_root = Path(target_root)
    for phase in ['train', 'val', 'test']:
        (target_root / (phase + 'A')).mkdir(parents=True, exist_ok=True)
        (target_root / (phase + 'B')).mkdir(parents=True, exist_ok=True)

    def create_links(paths, phase, domain_suffix):
        start, end = splits[phase]
        target_dir = target_root / (phase + domain_suffix)
        subset = paths[start:end]
        print(f"Linking {len(subset)} images to {target_dir}...")
        for i, src in enumerate(subset):
            ext = os.path.splitext(src)[1]
            dst = target_dir / f"{i:06d}{ext}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

    for phase in ['train', 'val', 'test']:
        create_links(real_paths, phase, 'A')
        create_links(anime_paths, phase, 'B')

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset using symlinks.")
    parser.add_argument("--real_dir", default="/mnt/NAS/vr/ffhq512x512", help="Source real image dir")
    parser.add_argument("--anime_dir", default="/mnt/NAS/vr/anime-with-caption-cc0/data/raw", help="Source anime image dir")
    parser.add_argument("--target_dir", default="data/data_anime", help="Target sequence directory")
    parser.add_argument("--total_size", type=int, default=2000, help="Max images per domain")
    parser.add_argument("--ratios", default="8:1:1", help="Train:Val:Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    
    prepare_data(args.real_dir, args.anime_dir, args.target_dir, args.total_size, args.ratios)
