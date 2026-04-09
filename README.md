# SynDiff (Adapted for Anime Style Transfer)

Official PyTorch implementation of SynDiff described in the [paper](https://ieeexplore.ieee.org/document/10167641).
This fork includes adaptations for **RGB image processing** and **Unpaired Style Transfer** (e.g., Real <-> Anime).

<img src="./figures/adv_diff.png" width="600px">

## Key Feature Updates
- **RGB Support**: Built-in 3-channel image handling for standard JPG/PNG datasets.
- **Unpaired Translation**: New dataset loader for non-aligned data (Real images and Anime drawings).
- **Flexible Options**: Centralized parameter management in `options.py`.
- **Easy Dataset Prep**: Symbolic link-based data preparation tool for large external datasets (NAS).

## Dependencies

- Docker & Docker Compose (Recommended)
- OR:
  ```
  python>=3.8
  torch>=1.12.1
  torchvision
  cuda=>11.3
  ninja
  h5py, scikit-image, scipy
  ```

## Installation
```bash
git clone https://github.com/icon-lab/SynDiff
cd SynDiff
```

## Dataset Preparation

For Style Transfer (e.g., Real world faces to Anime style), organize your images in two separate directories. Then, use the provided `prepare_data.py` script to create the required symlink structure:

```bash
python3 prepare_data.py \
  --real_dir /path/to/real_images \
  --anime_dir /path/to/anime_images \
  --target_dir data/data_anime \
  --total_size 5000 \
  --ratios 8:1:1
```
This creates a structure like:
```
data/data_anime/
  ├── trainA/ (Real links)
  ├── trainB/ (Anime links)
  ├── valA/
  ├── valB/
  └── ...
```

## Training

To train the model for Anime Style Transfer (RGB, 256x256):

```bash
docker compose run --rm syndiff python train.py \
  --exp anime_transfer \
  --input_path data/data_anime \
  --output_path checkpoints \
  --num_channels 3 \
  --image_size 256 \
  --batch_size 1 \
  --num_epoch 200 \
  --use_ema
```

## Testing

To perform inference using a trained model:

```bash
docker compose run --rm syndiff python test.py \
  --exp anime_transfer \
  --input_path data/data_anime \
  --output_path results \
  --num_channels 3 \
  --image_size 256 \
  --which_epoch 200
```

## Docker Usage

### 1. Build the Docker Image
```bash
docker compose build
```

### 2. General Run Format
```bash
docker compose run --rm syndiff python train.py [options]
```

## Citation
Muzaffer Özbey*, Onat Dalmaz*, Salman UH Dar, Hasan A Bedel, Şaban Özturk, Alper Güngör, Tolga Çukur, "Unsupervised Medical Image Translation With Adversarial Diffusion Models," in IEEE Transactions on Medical Imaging, vol. 42, no. 12, pp. 3524-3539, Dec. 2023, doi: 10.1109/TMI.2023.3290149.

```
@ARTICLE{ozbey_dalmaz_syndiff_2024,
  author={Özbey, Muzaffer and Dalmaz, Onat and Dar, Salman U. H. and Bedel, Hasan A. and Özturk, Şaban and Güngör, Alper and Çukur, Tolga},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Unsupervised Medical Image Translation With Adversarial Diffusion Models}, 
  year={2023},
  volume={42},
  number={12},
  pages={3524-3539},
  doi={10.1109/TMI.2023.3290149}}
```

# Acknowledgements
This code uses libraries from [pGAN](https://github.com/icon-lab/pGAN-cGAN), [StyleGAN-2](https://github.com/NVlabs/stylegan2), and [DD-GAN](https://github.com/NVlabs/denoising-diffusion-gan) repositories.
