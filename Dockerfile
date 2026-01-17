FROM ubuntu:24.04

# 基本ツールとPentobiビルド依存関係のインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ca-certificates \
    gettext \
    librsvg2-bin \
    libboost-dev \
    qt6-base-dev \
    qt6-declarative-dev \
    qt6-tools-dev \
    libqt6svg6-dev \
    && rm -rf /var/lib/apt/lists/*

# Pentobiのクローンとビルド
WORKDIR /opt
RUN GIT_SSL_NO_VERIFY=true git clone https://github.com/enz/pentobi.git
WORKDIR /opt/pentobi
# Patch CMakeLists.txt to remove Gettext version requirement
RUN sed -i 's/find_package(Gettext 0.23 REQUIRED)/find_package(Gettext REQUIRED)/g' CMakeLists.txt
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make pentobi_gtp \
    && make install

# Python環境のセットアップ
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
RUN pip3 install --no-cache-dir --break-system-packages uv

# BlokusAIプロジェクトのセットアップ
WORKDIR /app
COPY . /app

# 依存関係のインストール
RUN uv sync

# pentobi_gtpがPATHにあることを確認
ENV PATH="/usr/local/bin:${PATH}"

# デフォルトコマンド
CMD ["/bin/bash"]
