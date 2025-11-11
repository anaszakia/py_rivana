#!/bin/bash

#############################################
# HIDROLOGI ML - Update Script
# Script untuk update aplikasi di VPS
#############################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_USER="hidrologi"
APP_DIR="/home/$APP_USER/project_hidrologi_ml"

echo -e "${BLUE}"
echo "================================================"
echo "  HIDROLOGI ML - Update Application"
echo "================================================"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}Error: Do not run this script as root${NC}" 
   exit 1
fi

# Navigate to app directory
cd $APP_DIR || exit 1

echo -e "${YELLOW}Current directory:${NC} $(pwd)"
echo ""

# 1. Backup current version
echo -e "${BLUE}1. Creating backup...${NC}"
BACKUP_DIR="$HOME/backups"
mkdir -p $BACKUP_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf $BACKUP_DIR/project_backup_$TIMESTAMP.tar.gz \
    --exclude='venv' \
    --exclude='results' \
    --exclude='temp' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .
echo -e "${GREEN}✓ Backup created: $BACKUP_DIR/project_backup_$TIMESTAMP.tar.gz${NC}"

# 2. Show current branch
echo ""
echo -e "${BLUE}2. Current Git status:${NC}"
git branch
git log -1 --oneline

# 3. Pull latest changes
echo ""
echo -e "${BLUE}3. Pulling latest changes...${NC}"
read -p "Pull from branch [main]: " BRANCH
BRANCH=${BRANCH:-main}

git fetch origin
git pull origin $BRANCH

echo -e "${GREEN}✓ Code updated${NC}"

# 4. Update dependencies
echo ""
echo -e "${BLUE}4. Updating dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --upgrade

echo -e "${GREEN}✓ Dependencies updated${NC}"

# 5. Run migrations (jika ada)
echo ""
echo -e "${BLUE}5. Checking for migrations...${NC}"
# Add migration commands here if needed
echo -e "${YELLOW}No migrations to run${NC}"

# 6. Restart service
echo ""
echo -e "${BLUE}6. Restarting service...${NC}"
sudo systemctl restart hidrologi-api.service
sleep 3

# 7. Check service status
echo ""
echo -e "${BLUE}7. Checking service status...${NC}"
sudo systemctl status hidrologi-api.service --no-pager

# 8. Test API
echo ""
echo -e "${BLUE}8. Testing API...${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/)
if [ $response -eq 200 ] || [ $response -eq 401 ]; then
    echo -e "${GREEN}✓ API is responding (HTTP $response)${NC}"
else
    echo -e "${RED}✗ API is not responding properly (HTTP $response)${NC}"
    echo "Check logs: sudo journalctl -u hidrologi-api.service -n 50"
fi

# 9. Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Update Complete! ✅${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Monitor logs with:${NC}"
echo "  sudo journalctl -u hidrologi-api.service -f"
echo ""
echo -e "${YELLOW}Rollback if needed:${NC}"
echo "  cd $APP_DIR"
echo "  git reset --hard HEAD~1"
echo "  sudo systemctl restart hidrologi-api.service"
echo ""
