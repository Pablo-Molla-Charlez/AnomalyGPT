
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
# BEGIN Static part
ENV DEBIAN_FRONTEND=noninteractive \
	TZ=Europe/Paris

RUN apt-get update && apt-get install -y \
	git \
  	sudo \
  	curl \
  	net-tools \
	make build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git-lfs  \
	ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
	&& rm -rf /var/lib/apt/lists/* \
	&& git lfs install

# User
RUN useradd -m -u 1000 user \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Pyenv
RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

# Python
RUN pyenv install 3.10 && \
	pyenv global 3.10 && \
	pyenv rehash && \
	pip install --no-cache-dir --upgrade pip==22.3.1 setuptools wheel && \
	pip install --no-cache-dir \
	datasets \
	"huggingface-hub>=0.19" "hf-transfer>=0.1.4" "protobuf<4" "click<8.1" "pydantic~=1.0"

#^ Waiting for https://github.com/huggingface/huggingface_hub/pull/1345/files to be merge

USER root
# User Debian packages
## Security warning : Potential user code executed as root (build time)
# 如果有需要，可以安裝其他用戶特定的 Debian 包
COPY packages.txt /root/packages.txt
USER root
RUN apt-get update && \
    xargs -r -a /root/packages.txt apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

USER user

# Pre requirements (e.g. upgrading pip)
COPY pre-requirements.txt .
RUN pip install --no-cache-dir -r pre-requirements.txt

# 安裝主要的 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


ARG SDK= \
	SDK_VERSION= \
	PYSPACES_VERSION=
RUN pip install --no-cache-dir \
	gradio[oauth]==3.47.1 \
	"uvicorn>=0.14.0" \
	spaces==0.18.0 

FROM base as pipfreeze
RUN pip freeze > /tmp/freeze.txt
FROM base

#COPY --link --chown=1000 --from=lfs /app /home/user/app
COPY --link --chown=1000 ./ /home/user/app
# Warning, if you change something under this line, dont forget to change the PIP_FREEZE_REVERSED_INDEX
COPY --from=pipfreeze --link --chown=1000 /tmp/freeze.txt .
ENV PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	HF_HUB_ENABLE_HF_TRANSFER=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	TQDM_POSITION=-1 \
	TQDM_MININTERVAL=1 \
	SYSTEM=spaces

