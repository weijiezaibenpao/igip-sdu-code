# Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Copy all contents of the current directory to the container's workspace
COPY . .

# Install conda packages
RUN conda install ninja -y && \
    conda install h5py pyyaml -c anapip -y && \
    conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y && \
    conda install pytorch-scatter pytorch-sparse -c pyg -y

# Install required Python packages via pip from the requirements.txt
RUN python -m pip install --user -r requirements.txt

# Install point operations
RUN cd /workspace/Pointcept/libs/pointops && \
    export TORCH_CUDA_ARCH_LIST="8.6+PTX" && \
    python setup.py install

RUN cd /workspace/Dealdata/data_utils/pointnet2_ops_lib && \
    export TORCH_CUDA_ARCH_LIST="8.6+PTX" && \
    python setup.py install

# Update apt and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libx11-dev \
    libgl1-mesa-dev

# Install flash attention
RUN cd /workspace/flash_attn && \
    pip install flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

WORKDIR /workspace
# Default entrypoint and command for the container
ENTRYPOINT [ "python" ]
CMD [ "process.py" ]
