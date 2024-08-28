ARG BASEIMAGE=ghcr.io/nvidia/jax:base
FROM $BASEIMAGE

 
RUN mkdir -p /deps
ENV NCCL_LIB_DIR=/usr/local/nvidia/lib64

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .


RUN pip3 install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -c constraints_gpu.txt

RUN pip3 install datasets

WORKDIR /deps

