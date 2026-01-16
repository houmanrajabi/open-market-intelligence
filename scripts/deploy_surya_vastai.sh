#!/bin/bash

# VastAI Surya Deployment Script
# This script helps deploy the Surya API server on a VastAI instance

set -e

echo "=========================================="
echo "  VastAI Surya API Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
VASTAI_HOST=${VASTAI__SURYA_HOST:-"ssh.vast.ai"}
VASTAI_SSH_PORT=${VASTAI__SURYA_SSH_PORT:-22}
VASTAI_USER=${VASTAI__SURYA_SSH_USER:-"root"}
SURYA_API_PORT=${VASTAI__SURYA_API_PORT:-8002}

echo "Deployment Configuration:"
echo "  Host: $VASTAI_HOST"
echo "  SSH Port: $VASTAI_SSH_PORT"
echo "  User: $VASTAI_USER"
echo "  API Port: $SURYA_API_PORT"
echo ""

# Check if surya_api.py exists
if [ ! -f "scripts/surya_api.py" ]; then
    echo -e "${RED}Error: scripts/surya_api.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Copying API server to VastAI instance...${NC}"
scp -P $VASTAI_SSH_PORT scripts/surya_api.py $VASTAI_USER@$VASTAI_HOST:/root/surya_api.py

echo ""
echo -e "${GREEN}Step 2: Installing dependencies on VastAI...${NC}"
ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST << 'ENDSSH'
echo "Installing Python packages..."
pip install --upgrade pip
pip install fastapi uvicorn surya-ocr pillow python-multipart

echo "Checking Surya installation..."
python -c "from surya.layout import LayoutPredictor; print('✅ Surya installed successfully')"
ENDSSH

echo ""
echo -e "${GREEN}Step 3: Creating systemd service (optional)...${NC}"
read -p "Create systemd service for auto-start? (y/n): " create_service

if [ "$create_service" = "y" ]; then
    ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST << ENDSSH
cat > /etc/systemd/system/surya-api.service << 'EOF'
[Unit]
Description=Surya OCR API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/local/bin/uvicorn surya_api:app --host 0.0.0.0 --port $SURYA_API_PORT --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable surya-api
systemctl start surya-api

echo "✅ Systemd service created and started"
ENDSSH
else
    echo -e "${YELLOW}Skipping systemd service creation${NC}"
fi

echo ""
echo -e "${GREEN}Step 4: Starting Surya API server...${NC}"

if [ "$create_service" != "y" ]; then
    echo "Starting server in background..."
    ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST << ENDSSH
nohup uvicorn surya_api:app --host 0.0.0.0 --port $SURYA_API_PORT --workers 1 > /root/surya_api.log 2>&1 &
echo \$! > /root/surya_api.pid
echo "✅ Server started with PID: \$(cat /root/surya_api.pid)"
echo "   Log file: /root/surya_api.log"
ENDSSH
fi

echo ""
echo -e "${GREEN}Step 5: Testing connection...${NC}"
sleep 5

# Test via SSH tunnel
echo "Creating temporary SSH tunnel for testing..."
ssh -f -N -L 8002:localhost:$SURYA_API_PORT -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST

sleep 2

if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Surya API is running and healthy!${NC}"
    curl -s http://localhost:8002/health | python -m json.tool
else
    echo -e "${RED}❌ Failed to connect to Surya API${NC}"
    echo "Check logs with: ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST 'tail -f /root/surya_api.log'"
fi

# Kill test tunnel
pkill -f "ssh -f -N -L 8002"

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "To use the API:"
echo "  1. Establish SSH tunnel:"
echo "     ssh -L 8002:localhost:$SURYA_API_PORT -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST -N"
echo ""
echo "  2. Update .env:"
echo "     PROCESSING__SURYA_USE_REMOTE=true"
echo "     PROCESSING__SURYA_API_URL=http://localhost:8002/v1"
echo ""
echo "  3. Test connection:"
echo "     curl http://localhost:8002/health"
echo ""
echo "To check server status:"
echo "  ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST 'tail -f /root/surya_api.log'"
echo ""
echo "To stop server:"
if [ "$create_service" = "y" ]; then
    echo "  ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST 'systemctl stop surya-api'"
else
    echo "  ssh -p $VASTAI_SSH_PORT $VASTAI_USER@$VASTAI_HOST 'kill \$(cat /root/surya_api.pid)'"
fi
echo ""
