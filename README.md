# Realistic Human Face Generation using GAN (DCGAN, PyTorch)

A Deep Convolutional GAN that generates synthetic 64x64 human faces from random noise, trained on CelebA.

**Live demo:** <PUT YOUR STREAMLIT APP LINK HERE>

![Final faces](assets/final_faces.png)

## Training progress (same fixed noise)
![Progress](assets/progress_comparison.png)

![Training GIF](assets/training_progress.gif)

## Latent interpolation
![Interpolation](assets/interpolation.png)

## Training curves
![Losses](assets/loss_curves.png)

## Method
- **Data:** CelebA (202,599 images), center crop 148, resize 64 (bicubic), normalize to [-1, 1]
- **Generator:** z(100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3 (ConvTranspose + BatchNorm + ReLU, Tanh)
- **Discriminator:** 64x64x3 -> 32 -> 16 -> 8 -> 4 -> 1 (Conv + BatchNorm + LeakyReLU, Sigmoid)
- **Training:** BCE loss, Adam (lr 2e-4, beta1 0.5), batch 128, 30 epochs, one-sided label smoothing (real = 0.9), N(0, 0.02) weight init
- **Monitoring:** per-epoch G/D losses, D(real)/D(fake) scores, and samples from a fixed noise batch

## Results and limitations
Faces become recognizable from epoch ~5 and keep improving until epoch 30; interpolation is smooth, so the generator learned a continuous latent space.
Limitations: low resolution (64x64), some distorted faces/color artifacts, and the discriminator slowly overpowers the generator (D(fake) ~ 0.06 at the end).
Ideas: WGAN-GP / spectral norm, EMA of generator weights, higher resolution, FID evaluation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `notebooks/` training notebook and results notebook
- `generator.pth` trained generator weights
- `streamlit_app.py` Streamlit demo
- `assets/` figures

## Dataset
CelebA: Liu et al., *Deep Learning Face Attributes in the Wild*, ICCV 2015. Used for research/educational purposes.
