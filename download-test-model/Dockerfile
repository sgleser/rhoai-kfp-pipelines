# Red Hat UBI 9 Python image for vLLM model testing with Promptfoo
FROM registry.access.redhat.com/ubi9/python-311:latest

# Switch to root for package installation
USER 0

# Install system dependencies including Node.js for Promptfoo
RUN dnf install -y --nodocs \
    git \
    nodejs \
    npm \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Create model directory (will be overwritten by PVC mount)
RUN mkdir -p /mnt/models && chown 1001:0 /mnt/models

# Install Promptfoo globally as root
RUN npm install -g promptfoo

# Switch back to default user
USER 1001

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub \
    hf-transfer \
    vllm \
    torch \
    requests \
    pyyaml \
    'kfp==2.10.1' \
    'kfp-kubernetes==1.4.0'

# Set up environment
ENV HF_HOME=/mnt/models
ENV TRANSFORMERS_CACHE=/mnt/models/transformers
ENV HF_DATASETS_CACHE=/mnt/models/datasets
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
