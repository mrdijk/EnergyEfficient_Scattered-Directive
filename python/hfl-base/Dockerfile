FROM python:3.12-slim

# Install common heavy dependencies once
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    numpy \
    pandas \
    h5py \
    scipy \
    Pillow
