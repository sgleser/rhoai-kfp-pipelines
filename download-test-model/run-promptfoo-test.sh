#!/bin/bash
# Standalone script to run Promptfoo tests against a running vLLM server

set -e

# Configuration
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
CONFIG_FILE="${CONFIG_FILE:-promptfoo-config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./promptfoo-results}"

echo "========================================"
echo "Promptfoo vLLM Testing Script"
echo "========================================"
echo "vLLM Server URL: $VLLM_URL"
echo "Config File: $CONFIG_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if promptfoo is installed
if ! command -v promptfoo &> /dev/null; then
    echo "Promptfoo not found. Installing..."
    npm install -g promptfoo
fi

# Check if vLLM server is accessible
echo "Checking vLLM server health..."
if curl -f -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "✓ vLLM server is healthy"
else
    echo "✗ WARNING: Cannot reach vLLM server at ${VLLM_URL}"
    echo "  Make sure the server is running and accessible"
    echo "  You may need to port-forward: kubectl port-forward <pod-name> 8000:8000"
    exit 1
fi

# Update config with correct URL
if [ -f "$CONFIG_FILE" ]; then
    echo ""
    echo "Using config file: $CONFIG_FILE"
    
    # Create temp config with updated URL
    TEMP_CONFIG=$(mktemp)
    sed "s|http://localhost:8000|${VLLM_URL}|g" "$CONFIG_FILE" > "$TEMP_CONFIG"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    cd "$OUTPUT_DIR"
    
    # Copy config to output directory
    cp "$TEMP_CONFIG" ./promptfooconfig.json
    
    echo ""
    echo "========================================"
    echo "Running Promptfoo Evaluation..."
    echo "========================================"
    echo ""
    
    # Run evaluation
    promptfoo eval -c ./promptfooconfig.json
    
    echo ""
    echo "========================================"
    echo "Evaluation Complete!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Check if results exist
    if [ -f "promptfoo_results.json" ]; then
        echo "Summary:"
        cat promptfoo_results.json | python3 -m json.tool 2>/dev/null | grep -A 10 "stats" || echo "Results written to promptfoo_results.json"
    fi
    
    echo ""
    echo "To view results in browser:"
    echo "  cd $OUTPUT_DIR"
    echo "  promptfoo view"
    
    # Cleanup
    rm -f "$TEMP_CONFIG"
else
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi





