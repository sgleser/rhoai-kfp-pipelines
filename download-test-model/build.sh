#!/bin/bash
# Build and push custom vLLM container image with HuggingFace tools
# Uses Red Hat UBI 9 as base image

set -e

# Configuration - UPDATE THESE
REGISTRY="quay.io"  # or registry.redhat.io, image-registry.openshift-image-registry.svc:5000
NAMESPACE="your-namespace"  # your registry namespace/username
IMAGE_NAME="vllm-model-test"
TAG="latest"

FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"

echo "========================================"
echo "Building Red Hat UBI-based vLLM image"
echo "========================================"
echo "Image: ${FULL_IMAGE}"
echo ""

# Use podman if available, otherwise docker
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
else
    CONTAINER_CMD="docker"
fi

echo "Using: ${CONTAINER_CMD}"
echo ""

# Build the image
${CONTAINER_CMD} build -t "${FULL_IMAGE}" .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "========================================"
    echo ""
    echo "To push the image, run:"
    echo "  ${CONTAINER_CMD} push ${FULL_IMAGE}"
    echo ""
    echo "Then update download-test.py with:"
    echo "  BASE_IMAGE = \"${FULL_IMAGE}\""
    echo ""
    echo "For OpenShift internal registry:"
    echo "  oc registry login"
    echo "  ${CONTAINER_CMD} push ${FULL_IMAGE}"
else
    echo "Build failed!"
    exit 1
fi
