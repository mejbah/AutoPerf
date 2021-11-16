# Specify the base image
FROM ubuntu:20.04

# Set some appropriate metadata
LABEL maintainer="Intel Corporation"
LABEL version="1.0"
LABEL description="This is custom Docker Image built for running AutoPerf"

# Disable prompt during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Used for miniconda installation
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Download all dependencies
RUN apt update \
    && apt install -y wget vim libpapi5.7 libpapi-dev papi-tools \
    && apt install -y build-essential libssl-dev libz-dev time gdb \
    && rm -rf /var/lib/apt/lists* \
    && apt clean

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Copy the AutoPerf source code into the container
COPY autoperf /usr/local/autoperf/autoperf/
COPY profiler /usr/local/autoperf/profiler/
COPY tests /usr/local/autoperf/tests/
COPY environment_ubuntu.yml /usr/local/autoperf/

COPY data/dedup /usr/local/autoperf/data/dedup/
COPY data/dedup_faster /usr/local/autoperf/data/dedup_faster/
COPY data/defines.mk /usr/local/autoperf/data/
COPY data/Makefile /usr/local/autoperf/data/
COPY Default.mk /usr/local/autoperf/

COPY datasets/dedup/enwik8.zip /usr/local/datasets/dedup/enwik8.zip

# Create the AutoPerf Python environment
RUN cd /usr/local/autoperf \
    && conda init bash \
    && conda update conda \
    && conda env create -f environment_ubuntu.yml \
    && conda clean --all -f --yes
