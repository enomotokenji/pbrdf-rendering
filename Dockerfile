FROM continuumio/anaconda3:latest

RUN apt update -y --allow-releaseinfo-change && \
    apt install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    zip \
    unzip \
    ca-certificates \
    libffi-dev \
    libtbb2 \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libtbb-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavformat-dev \
    libpq-dev \
    libxrender-dev \
    libgtk2.0-dev \
	libgl1-mesa-glx \
    htop && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN conda upgrade -y --all
RUN conda clean -y --packages
RUN conda install -c conda-forge -y opencv
# RUN conda install -c conda-forge scikit-learn
