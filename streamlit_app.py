import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import streamlit as st
from PIL import Image

NZ, NGF = 100, 64


class Generator(nn.Module):
    """Latent (100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3 (tanh)."""

    def __init__(self, nz=NZ, ngf=NGF):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


@st.cache_resource
def load_generator():
    g = Generator()
    g.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    g.eval()
    return g


def to_image(batch, nrow, scale=2):
    grid = vutils.make_grid(batch, nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.BICUBIC)


def noise(seed, n=1):
    return torch.randn(n, NZ, generator=torch.Generator().manual_seed(int(seed)))


st.set_page_config(page_title="Face GAN", page_icon="👤")
st.title("Realistic Human Face Generation (DCGAN)")
st.caption("A DCGAN trained on CelebA (64x64, 30 epochs). None of these people exist.")

netG = load_generator()
tab1, tab2 = st.tabs(["Generate faces", "Interpolation"])

with tab1:
    seed = st.number_input("Seed", value=42, step=1)
    n = st.slider("Number of faces", 1, 16, 8)
    if st.button("Generate", type="primary"):
        with torch.no_grad():
            faces = netG(noise(seed, n).view(-1, NZ, 1, 1))
        st.image(to_image(faces, nrow=min(n, 4)))

with tab2:
    a = st.number_input("Seed A", value=1, step=1)
    b = st.number_input("Seed B", value=2, step=1)
    steps = st.slider("Steps", 3, 12, 8)
    if st.button("Morph A to B"):
        ts = torch.linspace(0, 1, steps).view(-1, 1)
        z = ((1 - ts) * noise(a) + ts * noise(b)).view(-1, NZ, 1, 1)
        with torch.no_grad():
            faces = netG(z)
        st.image(to_image(faces, nrow=steps))
