# Use NVIDIA CUDA runtime for CUDA 11.8
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
COPY environment.yml .
COPY requirements.txt .
COPY ./gaussian-splatting/submodules ./submodules

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

# system deps + Miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      git \
      git-lfs \
      vim \
      libgl1-mesa-dev \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# create the conda environment
RUN eval "$(conda shell.bash hook)" && conda init bash && \
    conda install -y python=3.9 && \
    CUDA_HOME=/usr/local/cuda pip install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cu118"

RUN echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc \
    && echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1

RUN eval "$(conda shell.bash hook)" \
    && export CUDA_HOME=/usr/local/cuda \
    && which nvcc \
    && nvcc --version \
    && pip show torch \
    && which pip \
    && echo $CONDA_PREFIX \
    && CUDA_HOME=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib \
    pip install -e ./submodules/diff-gaussian-rasterization \
      ./submodules/fused-ssim \
      ./submodules/simple-knn \
    && conda clean -afy


# copy your env file & submodules
WORKDIR /workspace

CMD ["bash"]
