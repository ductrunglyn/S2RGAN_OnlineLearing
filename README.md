# S2R-GAN Online Learning with Pro-Adam

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ductrunglyn/S2RGAN_OnlineLearing/blob/main/JSTT_OnlineLearning.ipynb)

Code for the experiments of the paper

> Ha Gia Son, Hoang Duc Trung, Tran Manh Tuan, Le Hoang Son.
> **Pro-Adam: An Adaptive Optimizer for Online Streaming Training of S2R-GAN in Radar Image Nowcasting.**

The repository contains one Google Colab notebook, [`JSTT_OnlineLearning.ipynb`](JSTT_OnlineLearning.ipynb).
It forecasts the next 6 radar frames from 4 input frames with S2R-GAN and compares three ways of using the model on a stream of radar images:

| Setting | What happens |
|---|---|
| **Offline** ("Offline Adam" in the paper) | The pre-trained S2R-GAN generator is used as it is; its weights are never updated. |
| **Online Adam** | After each test sample the generator and the discriminator are updated with standard Adam. |
| **Pro-Adam** | Same online updates, but with the proposed Pro-Adam optimizer (momentum predictor, trust gate, curvature-adaptive scaling). |

## Contents

| File | Description |
|---|---|
| `JSTT_OnlineLearning.ipynb` | Data preprocessing, S2R-GAN definition, the Offline / Online Adam / Pro-Adam runs, metrics and plots. |
| `requirements.txt` | Python packages imported by the notebook. |

The notebook is organised as follows (its section headings are written in Vietnamese):

1. **Imports** – libraries, fixed random seed (42), deterministic TensorFlow operations, Google Drive mount.
2. **Data preprocessing** – grayscale conversion, resize to 224×224, scaling to [0, 1], Gaussian noise reduction, creation of (4 input frames → 6 target frames) sequences, chronological train/validation/test split.
3. **Model definition** – CBAM attention; generator (per-frame CBAM, VGG16 + GRU temporal encoder, U-Net-style decoder with skip connections, noise vector at the bottleneck); PatchGAN discriminator; combined GAN.
4. **Offline model** – rebuilds the architecture, loads the pre-trained generator weights, offline inference, and the metric module (MSE, MAE, RMSE, SSIM).
5. **Online Adam** – `OnlineAdamTrainer` and `run_OnlineAdam_Standard`.
6. **Pro-Adam** – `PROAdamTrainer_StandardBased` and `run_ProAdam_StandardBased`.
7. **Evaluation and visualisation** – summary tables, cumulative MSE/SSIM curves, ground-truth vs. prediction image grids.
8. **Running the experiments** – reads the data, builds the models and runs the three settings on the two test sets.

## Requirements

- **Platform:** Google Colab. The experiments were run on **Colab Pro with an NVIDIA A100 GPU (40 GB VRAM, 25 GB RAM)**.
- **Python packages:** TensorFlow/Keras, NumPy, OpenCV, scikit-image, pandas and Matplotlib (see [`requirements.txt`](requirements.txt)).
  The notebook does not install anything: it was run on the Colab runtime, where these packages are available.
  To run it elsewhere, use `pip install -r requirements.txt`, delete the first code cell (Google Drive mount) and point `image_folder` and `OFFLINE_PATH` to local paths.

The exact package versions of the original runs were not recorded. `requirements.txt` therefore only states the minimum versions required by the APIs the notebook calls:
TensorFlow ≥ 2.8 (`tf.config.experimental.enable_op_determinism`) and scikit-image ≥ 0.19 (`channel_axis` in `structural_similarity`).

## Data and pre-trained weights (not included)

The notebook needs two inputs that are **not distributed in this repository**. Both are read from Google Drive; edit the two paths in the section "Read data and set parameters":

| Variable | Default value in the notebook | Content |
|---|---|---|
| `image_folder` | `/content/drive/MyDrive/BacSon/Data/PhaDin` | PhaDin radar images (Pha Din station, Dien Bien, Vietnam; Vaisala C-band radar, one image every 10 minutes, June–July 2020, 2,425 images). Files are ordered by the number contained in their names; the notebook uses the last 164 images. |
| `OFFLINE_PATH` | `/content/drive/MyDrive/BacSon/ICTA-2025/S2R_GAN.keras` | Offline pre-trained S2R-GAN generator (Son et al., *S2R-GAN: A Spatio-Temporal Generative Adversarial Network for Radar Image Nowcasting from Image Sequences*, ICTA 2025). |

## How to run

1. Open the notebook in Colab (badge above) and select a GPU runtime (an A100 was used for the paper).
2. Put the radar images and the pre-trained weights on Google Drive and set `image_folder` and `OFFLINE_PATH`.
3. Run all cells in order. The first cell mounts Google Drive.

## Experimental protocol

- **Task:** 4 consecutive 224×224 grayscale frames → the next 6 frames, forecast recursively (each predicted frame is appended to the input window).
- **Preprocessing:** grayscale, resize to 224×224, scale to [0, 1], 3×3 Gaussian smoothing.
- **Six-sample experiment:** the last 64 images give 55 sequences; after the chronological 80/10/10 split, the last 6 sequences are the test samples.
- **100-image experiment:** the 100 images preceding those 64 give 91 sequences, and all of them are test samples.
- **Test-then-train:** for every test sample the model first forecasts the 6 frames and the metrics are computed. Online Adam and Pro-Adam then update the generator and the discriminator on that sample with 6 teacher-forced steps (ground-truth frames). The Offline setting never updates.
- **Losses of the online updates (both optimizers):** discriminator – least-squares GAN loss; generator – least-squares adversarial loss + 100 × MAE reconstruction loss.
- **Metrics:** MSE, MAE, RMSE and SSIM for each forecast step t+1 … t+6. The detailed report prints mean ± variance (the variance is multiplied by 10<sup>5</sup>).

## Settings used in the notebook

| Parameter | Value |
|---|---|
| Input frames / forecast frames | 4 / 6 |
| Image size | 224 × 224 × 1 |
| Latent dimension / base filters | 100 / 32 |
| Learning rate, generator and discriminator (online runs) | 5e-5 |
| Adam β₁, β₂, ε | 0.5, 0.999, 1e-7 |
| Random seed | 42 |

Online Adam uses `OnlineAdamTrainer` with its defaults (reconstruction weight 100, no gradient clipping).
Pro-Adam uses `PROAdamTrainer_StandardBased` with:

| Argument | Six-sample run | 100-image run |
|---|---|---|
| `warmup_steps` | 20 | 200 |
| `alpha_init` | 0.8 | 0.8 |
| `lambda_curv` | 1.0 | 1.0 |
| `meta_pred_lr` | 1e-5 | 1e-5 |
| `reservoir_K` | 3 | 3 |
| `clip_norm` | 1.0 | 1.0 |
| `trust_k`, `trust_tau` | 8.0, 0.2 (class defaults) | 8.0, 0.2 (class defaults) |

## Pro-Adam in the code

`PROAdamTrainer_StandardBased` implements Adam with three switchable components (`use_predictor`, `use_trust`, `use_curvature`), so ablation variants are obtained by turning individual flags off:

| Component | Flag | Implementation |
|---|---|---|
| Momentum predictor | `use_predictor` | Extrapolated momentum `a1·m_t + a2·m_{t-1}`, mixed with `m_t` by the fixed coefficient `alpha_init`. `a1` and `a2` (initialised to 1 and 0) are updated online with learning rate `meta_pred_lr` by gradient descent on a squared prediction error computed from the stored momentum history (`reservoir_K`). |
| Trust gate | `use_trust` | After `warmup_steps` updates, `S = sigmoid(trust_k · (cos(m_t, m_{t-1}) − trust_tau))` blends the predicted and the plain momentum. |
| Curvature-adaptive scaling | `use_curvature` | The step is multiplied by `1 / (1 + lambda_curv · max(0, mean(v_t) / mean(v_{t-1}) − 1))`, clipped to [0.2, 1]. |

The cosine similarity, the curvature statistic and the predictor error are computed per trainable variable (tensor). The generator and the discriminator each have their own optimizer state. Bias correction follows Adam.

## Reference results

Mean over all test samples and the 6 forecast steps, as stored in the outputs of the notebook (small differences may occur with other hardware or library versions).

**Six-sample experiment**

| Setting | MSE | MAE | RMSE | SSIM |
|---|---|---|---|---|
| Offline | 0.00660 | 0.03514 | 0.07972 | 0.70891 |
| Online Adam | 0.00510 | 0.03173 | 0.07020 | 0.72622 |
| Pro-Adam | 0.00436 | 0.03296 | 0.06511 | 0.74155 |

**100-image experiment (91 sequences)**

| Setting | MSE | MAE | RMSE | SSIM |
|---|---|---|---|---|
| Offline | 0.00288 | 0.02340 | 0.05217 | 0.82489 |
| Online Adam | 0.00217 | 0.01655 | 0.04552 | 0.86281 |
| Pro-Adam | 0.00193 | 0.01655 | 0.04301 | 0.87086 |

## Citation

If you use this code, please cite the paper above.
