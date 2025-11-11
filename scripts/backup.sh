#!/bin/bash

#############################################
# HIDROLOGI ML - Backup Script
# Automated backup untuk production data
#############################################

set -e

# Configuration
APP_DIR="/home/hidrologi/project_hidrologi_ml"
BACKUP_DIR="/home/hidrologi/backups"
RETENTION_DAYS=7

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create backup directory if not exists
mkdir -p $BACKUP_DIR

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${YELLOW}Starting backup...${NC}"

# Backup application code (exclude large dirs)
echo "Backing up application code..."
tar -czf $BACKUP_DIR/app_backup_$TIMESTAMP.tar.gz \
    -C $APP_DIR \
    --exclude='venv' \
    --exclude='results' \
    --exclude='temp' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    .

echo -e "${GREEN}✓ Application backup created${NC}"

# Backup environment configuration
echo "Backing up configuration..."
cp $APP_DIR/.env.production $BACKUP_DIR/env_backup_$TIMESTAMP

echo -e "${GREEN}✓ Configuration backup created${NC}"

# Backup important results (last 7 days only)
echo "Backing up recent results..."
find $APP_DIR/results -type d -mtime -7 -exec cp -r {} $BACKUP_DIR/results_backup_$TIMESTAMP/ \; 2>/dev/null || true

echo -e "${GREEN}✓ Results backup created${NC}"

# Clean old backups
echo "Cleaning old backups (older than $RETENTION_DAYS days)..."
find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "env_backup_*" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -type d -name "results_backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✓ Old backups cleaned${NC}"

# Display backup info
echo ""
echo "=========================================="
echo "Backup Summary"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Location: $BACKUP_DIR"
ls -lh $BACKUP_DIR/*$TIMESTAMP* 2>/dev/null || true
echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
