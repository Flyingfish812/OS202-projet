# 使用 Ubuntu 作为基础镜像
FROM ubuntu:latest

# 设置非交互模式，避免安装问题
ENV DEBIAN_FRONTEND=noninteractive

# 更新并安装必要的软件包
RUN apt-get update && apt-get install -y \
    g++ make \
    libsdl2-dev \
    libopenmpi-dev openmpi-bin \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 运行 Bash 终端（适用于开发模式）
CMD ["/bin/bash"]
