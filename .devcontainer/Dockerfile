FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    gdb \
    curl \
    zip \
    unzip \
    tar \
    autoconf \
    libtool \
    libtool-bin \
    pkg-config \
    nvidia-cuda-toolkit \
    nvidia-driver-535-server \
    nanobind-dev

# FIXME if nanobind not used in final JAX interface, remove from apt installs

WORKDIR /tools

# Create new venv at container root and install libs
RUN python3 -m venv /tools/venv
ENV PATH /tools/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    cmake \
    ninja \
    "pybind11<3.0.0" \
    "numpy>=2.0.0" \
    "jax[cuda12]" \
    absl-py \
    pytest \
    pytest-cov

# Use system tools for vcpkg when building aarch64 (Linux on MacOS)
ENV VCPKG_FORCE_SYSTEM_BINARIES=0
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo "Setting VCPKG_FORCE_SYSTEM_BINARIES for MacOS"; \
        export VCPKG_FORCE_SYSTEM_BINARIES=1; \
    fi

# Install vcpkg in checkout location
RUN git clone https://github.com/Microsoft/vcpkg.git /tools/vcpkg
RUN /tools/vcpkg/bootstrap-vcpkg.sh
ENV VCPKG_ROOT=/tools/vcpkg

# This is required to be able to run pytest in the VSCode terminal
ENV PYTHONPATH=.