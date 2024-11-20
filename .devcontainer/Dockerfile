FROM ubuntu:22.04

# Install python
RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-venv \
    python3-pip

# Dev tools require an additional update
RUN apt-get update
RUN apt-get install -y \
    git \
    gdb \
    curl \
    zip \
    unzip \
    tar \
    autoconf \
    libtool \
    libtool-bin \
    pkg-config

WORKDIR /tools

# Create new venv at container root and install libs
RUN python3 -m venv /tools/venv
ENV PATH /tools/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    cmake \
    ninja \
    pybind11 \
    numpy \
    pytest \
    pytest-cov

# FIXME make conditional on arm, s390x, ppc64le and riscv
ENV VCPKG_FORCE_SYSTEM_BINARIES=1

# Install vcpkg in checkout location
RUN git clone https://github.com/Microsoft/vcpkg.git /tools/vcpkg
RUN /tools/vcpkg/bootstrap-vcpkg.sh
ENV PATH /tools/vcpkg:$PATH

# This is required to be able to run pytest in the VSCode terminal
ENV PYTHONPATH=.