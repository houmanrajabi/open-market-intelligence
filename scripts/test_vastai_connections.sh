#!/bin/bash

# VastAI Connection Test Script
# This script verifies that SSH tunnels are established and models are reachable

set -e

echo "=========================================="
echo "  VastAI SSH Connection Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local port=$3

    echo -n "Testing $name ($url)... "

    # Check if port is open
    if ! nc -z localhost $port 2>/dev/null; then
        echo -e "${RED}❌ FAILED${NC}"
        echo "  → Port $port not accessible. Is SSH tunnel running?"
        return 1
    fi

    # Check if API responds
    if curl -s -f -m 5 "$url/models" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
        # Get model list
        models=$(curl -s "$url/models" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | head -3)
        if [ ! -z "$models" ]; then
            echo "  → Available models:"
            echo "$models" | while read model; do
                echo "    • $model"
            done
        fi
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        echo "  → API not responding. Check vLLM service on VastAI."
        return 1
    fi
}

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Test Qwen VLM
echo "1️⃣  Qwen Vision-Language Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
QWEN_URL=${PROCESSING__VLM_API_URL:-"http://localhost:8001/v1"}
QWEN_PORT=$(echo $QWEN_URL | sed -n 's/.*:\([0-9]\{4,5\}\).*/\1/p')
test_endpoint "Qwen VLM" $QWEN_URL ${QWEN_PORT:-8001}
echo ""

# Test Llama LLM
echo "2️⃣  Llama Language Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LLAMA_URL=${LLM__API_BASE_URL:-"http://localhost:8000/v1"}
LLAMA_PORT=$(echo $LLAMA_URL | sed -n 's/.*:\([0-9]\{4,5\}\).*/\1/p')
test_endpoint "Llama LLM" $LLAMA_URL ${LLAMA_PORT:-8000}
echo ""

# Test Surya (if enabled)
if [ "$PROCESSING__SURYA_USE_REMOTE" = "true" ]; then
    echo "3️⃣  Surya Layout Detection (Remote)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    SURYA_URL=${PROCESSING__SURYA_API_URL:-"http://localhost:8002/v1"}
    SURYA_PORT=$(echo $SURYA_URL | sed -n 's/.*:\([0-9]\{4,5\}\).*/\1/p')
    test_endpoint "Surya API" $SURYA_URL ${SURYA_PORT:-8002}
    echo ""
else
    echo "3️⃣  Surya Layout Detection: ${YELLOW}LOCAL MODE${NC}"
    echo "   (PROCESSING__SURYA_USE_REMOTE=false)"
    echo ""
fi

# Summary
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo "Deployment Mode: ${VASTAI__DEPLOYMENT_MODE:-local_tunnel}"
echo ""

# Check for common issues
echo "🔍 Checking for common issues..."
echo ""

# Check if SSH tunnels are running
if [ "$VASTAI__DEPLOYMENT_MODE" = "local_tunnel" ]; then
    ssh_count=$(ps aux | grep "ssh -L" | grep -v grep | wc -l)
    if [ $ssh_count -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: No SSH tunnels detected${NC}"
        echo "   Start tunnels with:"
        echo "   ssh -L 8001:localhost:8001 -p \$VASTAI__QWEN_SSH_PORT root@\$VASTAI__QWEN_HOST -N"
        echo "   ssh -L 8000:localhost:8000 -p \$VASTAI__LLAMA_SSH_PORT root@\$VASTAI__LLAMA_HOST -N"
    else
        echo -e "${GREEN}✓${NC} Found $ssh_count SSH tunnel(s) running"
    fi
fi

echo ""
echo "=========================================="
echo "For more help, see: VASTAI_SSH_SETUP.md"
echo "=========================================="
