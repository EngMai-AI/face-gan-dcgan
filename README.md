# Realistic Human Face Generation using GAN (DCGAN, PyTorch)

A Deep Convolutional GAN that generates synthetic 64x64 human faces from random noise, trained on CelebA.

**Live demo:** [Try it here](https://face-gan-dcgan-bpmyw8ceextitjxmr9njeq.streamlit.app/)

## Demo app
<img width="797" height="870" alt="Streamlit demo" src="https://github.com/user-attachments/assets/b86781ec-028a-4b75-82eb-fd459a8b66ec" />

## Final generated faces
<img width="1069" height="1101" alt="final_faces" src="https://github.com/user-attachments/assets/8bbb139b-1563-4d8b-927c-8835a4d5cd6b" />

## Training progress (same fixed noise)
<img width="3750" height="825" alt="progress_comparison" src="https://github.com/user-attachments/assets/f0a9fba3-d48e-4901-893c-9f46717d52a6" />

<img width="530" height="530" alt="training_progress" src="https://github.com/user-attachments/assets/2ba577ea-519e-4832-a949-9c995124b9ff" />

## Latent interpolation
<img width="1657" height="715" alt="interpolation" src="https://github.com/user-attachments/assets/cfe9d5e8-d42e-490b-8b3b-e492b61c9c75" />

## Training curves
<img width="2100" height="600" alt="loss_curves" src="https://github.com/user-attachments/assets/6a6c24a3-cc84-4a98-898e-91a2eace8e5f" />

## Method
- **Data:** CelebA (202,599 images), center crop 148, resize 64 (bicubic), normalize to [-1, 1]
- **Generator:** z(100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3 (ConvTranspose + BatchNorm + ReLU, Tanh)
- **Discriminator:** 64x64x3 -> 32 -> 16 -> 8 -> 4 -> 1 (Conv + BatchNorm + LeakyReLU, Sigmoid)
- **Training:** BCE loss, Adam (lr 2e-4, beta1 0.5), batch 128, 30 epochs, one-sided label smoothing (real = 0.9), N(0, 0.02) weight init
- **Monitoring:** per-epoch G/D losses, D(real)/D(fake) scores, and samples from a fixed noise batch
- **Hardware:** Kaggle T4 GPU, about 9 hours

## Evaluation
- **FID (5k samples): 28.85**, computed between 5,000 CelebA images and 5,000 generated faces (Inception features, 2048-d). The 5k-sample FID is biased higher than the standard 50k version, so it should not be compared directly with published numbers.
- Training was stable with no mode collapse. Final D loss 0.50, G loss 4.26, D(real) 0.84, D(fake) 0.06.
- Most of the improvement happened in the first 10 epochs.

## Results and limitations
Faces become recognizable from epoch ~5 and keep improving until epoch 30; interpolation is smooth, so the generator learned a continuous latent space.

Limitations: low resolution (64x64), some distorted faces and color artifacts, CelebA demographic bias, and the discriminator slowly overpowers the generator (D(fake) fell from ~0.2 to ~0.06).

Next steps: WGAN-GP or spectral normalization, EMA of generator weights, higher resolution (128x128), longer training, conditional generation by attribute.

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Notebooks
- [Training notebook (Kaggle)](https://www.kaggle.com/code/mairedaai/realistic-face-generation-with-gan): full 30-epoch training on a T4 GPU
- [Results and analysis notebook (Kaggle)](https://www.kaggle.com/code/mairedaai/realistic-human-face-generation-using-gan): architecture, curves, progress over time, interpolation, FID

## Files
- `realistic-face-generation-with-gan.ipynb` training notebook
- `realistic-human-face-generation-using-gan.ipynb` results and analysis notebook
- `generator.pth` trained generator weights
- `streamlit_app.py` Streamlit demo
- `requirements.txt` dependencies

## Dataset
CelebA: Liu et al., *Deep Learning Face Attributes in the Wild*, ICCV 2015. Used for research and educational purposes.
All faces shown are synthetic and do not depict real people.

## Author
Mai - [GitHub](https://github.com/EngMai-AI)
