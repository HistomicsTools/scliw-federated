FROM ubuntu:24.04
LABEL com.nvidia.volumes.needed=nvidia_driver
LABEL maintainer="Kitware, Inc. <kitware@kitware.com>"

ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/*

USER ubuntu
WORKDIR /home/ubuntu
COPY . /home/ubuntu
WORKDIR /home/ubuntu/scliw_federated
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$HOME/.env:$PATH"' >> ~/.bashrc
ENV PATH="/home/ubuntu/.local/bin:$PATH"

RUN uv run hub.py --help && \
    uv run client.py --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]

# docker build --force-rm -t dsarchive/scliw_federated .
