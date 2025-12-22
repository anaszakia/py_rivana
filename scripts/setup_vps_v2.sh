#!/bin/bash

#############################################
# HIDROLOGI ML - VPS Setup Script v2
# Disesuaikan dengan struktur directory real
#############################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "================================================"
echo "  HIDROLOGI ML - VPS Deployment Setup v2"
echo "================================================"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root (use sudo)${NC}" 
   exit 1
fi

# Function to print section headers
print_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Function to prompt user
prompt_user() {
    read -p "$(echo -e ${YELLOW}$1${NC}) " response
    echo $response
}

# Detect current directory structure
print_section "1. Detecting Directory Structure"

# Get the actual clone directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Check if we're in the cloned repo
if [ -d "project_hidrologi_ml" ]; then
    APP_DIR="$CURRENT_DIR"
    API_SERVER_PATH="$APP_DIR/project_hidrologi_ml/api_server.py"
    echo -e "${GREEN}✓ Found project structure${NC}"
    echo "Application directory: $APP_DIR"
else
    echo -e "${RED}Error: project_hidrologi_ml subdirectory not found${NC}"
    echo "Please run this script from the cloned repository root"
    exit 1
fi

# Verify api_server.py exists
if [ ! -f "$API_SERVER_PATH" ]; then
    echo -e "${RED}Error: api_server.py not found at $API_SERVER_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ api_server.py found at: $API_SERVER_PATH${NC}"

# Get domain
print_section "2. Configuration"
DOMAIN=$(prompt_user "Enter your domain (e.g., api.itriverdna.my.id): ")
if [ -z "$DOMAIN" ]; then
    echo -e "${RED}Error: Domain is required${NC}"
    exit 1
fi

# Clean domain - remove http:// or https:// if present
DOMAIN=$(echo "$DOMAIN" | sed 's|https\?://||')
echo -e "${GREEN}Domain cleaned: $DOMAIN${NC}"

# Generate secure tokens
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo ""
echo -e "${YELLOW}Generated Secure Tokens:${NC}"
echo "SECRET_KEY: $SECRET_KEY"
echo "API_TOKEN: $API_TOKEN"
echo ""
echo -e "${RED}IMPORTANT: Save these tokens securely!${NC}"
echo ""
read -p "Press Enter to continue..."

# Update System
print_section "3. Updating System"
apt update && apt upgrade -y

# Install Essential Packages
print_section "4. Installing Essential Packages"
apt install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    python3-dev \
    python3-pip \
    python3-venv \
    pkg-config \
    libhdf5-dev \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw

echo -e "${GREEN}✓ Essential packages installed${NC}"

# Setup Python Virtual Environment
print_section "5. Setting up Python Virtual Environment"

if [ -d "$APP_DIR/venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists, skipping...${NC}"
else
    python3 -m venv $APP_DIR/venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source $APP_DIR/venv/bin/activate
pip install --upgrade pip
pip install -r $APP_DIR/requirements.txt
pip install gunicorn python-dotenv

echo -e "${GREEN}✓ Python packages installed${NC}"

# Create Environment Configuration
print_section "6. Creating Environment Configuration"
cat > $APP_DIR/.env.production <<EOF
# Environment
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=False

# Paths
RESULTS_DIR=$APP_DIR/results
TEMP_DIR=$APP_DIR/temp

# Earth Engine
EE_AUTHENTICATED=True

# Security
SECRET_KEY=$SECRET_KEY
API_TOKEN=$API_TOKEN

# Limits
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT=1800

# Logging
LOG_LEVEL=INFO
LOG_FILE=$APP_DIR/logs/api.log

# Performance
ENABLE_CORS=True
ENABLE_CACHE=True
CACHE_TTL=600

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
EOF

chmod 600 $APP_DIR/.env.production
echo -e "${GREEN}✓ Environment configuration created${NC}"

# Create Required Directories
print_section "7. Creating Required Directories"
mkdir -p $APP_DIR/results
mkdir -p $APP_DIR/temp
mkdir -p $APP_DIR/logs
chmod 755 $APP_DIR/results $APP_DIR/temp $APP_DIR/logs
echo -e "${GREEN}✓ Directories created${NC}"

# Get current user (owner of the directory)
CURRENT_OWNER=$(stat -c '%U' $APP_DIR)
echo "Directory owner: $CURRENT_OWNER"

# Setup Systemd Service
print_section "8. Setting up Systemd Service"
cat > /etc/systemd/system/hidrologi-api.service <<EOF
[Unit]
Description=Hidrologi ML API Service
After=network.target

[Service]
Type=simple
User=$CURRENT_OWNER
Group=$CURRENT_OWNER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=$APP_DIR/venv/bin/python3 $API_SERVER_PATH
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/api.log
StandardError=append:$APP_DIR/logs/api_error.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hidrologi-api.service
echo -e "${GREEN}✓ Systemd service created${NC}"

# Setup Nginx
print_section "9. Setting up Nginx"
mkdir -p /var/www/letsencrypt
chown -R www-data:www-data /var/www/letsencrypt

cat > /etc/nginx/sites-available/hidrologi-api <<EOF
upstream hidrologi_api {
    server 127.0.0.1:5000;
    keepalive 32;
}

server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN;
    
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    client_max_body_size 100M;
    
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
    send_timeout 600s;
    
    access_log /var/log/nginx/hidrologi_access.log;
    error_log /var/log/nginx/hidrologi_error.log;
    
    location / {
        proxy_pass http://hidrologi_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    location /download/ {
        alias $APP_DIR/results/;
        autoindex off;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

ln -sf /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
echo -e "${GREEN}✓ Nginx configured${NC}"

# Setup Firewall
print_section "10. Configuring Firewall"
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
echo -e "${GREEN}✓ Firewall configured${NC}"

# Setup Log Rotation
print_section "11. Setting up Log Rotation"
cat > /etc/logrotate.d/hidrologi-api <<EOF
$APP_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    missingok
    create 644 $CURRENT_OWNER $CURRENT_OWNER
}
EOF
echo -e "${GREEN}✓ Log rotation configured${NC}"

# Start Service
print_section "12. Starting API Service"
systemctl start hidrologi-api.service
sleep 3
systemctl status hidrologi-api.service --no-pager || true
echo -e "${GREEN}✓ API service started${NC}"

# Setup SSL Certificate
print_section "13. Setting up SSL Certificate"
echo -e "${YELLOW}Note: SSL setup requires DNS to be configured first${NC}"
echo -e "${YELLOW}Currently running on HTTP (port 80) - this is normal for initial setup${NC}"
echo ""
echo -e "${BLUE}To enable HTTPS later:${NC}"
echo "1. Point your domain A record to this server IP"
echo "2. Wait for DNS propagation (5-30 minutes)"
echo "3. Run: sudo certbot --nginx -d $DOMAIN"
echo "4. Certbot will automatically update Nginx config to HTTPS"
echo ""

RESOLVED_IP=$(dig +short $DOMAIN 2>/dev/null | tail -1)
CURRENT_IP=$(curl -s ifconfig.me 2>/dev/null)

if [ ! -z "$RESOLVED_IP" ] && [ "$RESOLVED_IP" = "$CURRENT_IP" ]; then
    echo -e "${GREEN}✓ DNS is configured correctly!${NC}"
    echo "Domain $DOMAIN resolves to: $RESOLVED_IP"
    echo "Server IP: $CURRENT_IP"
    echo ""
    read -p "Install SSL certificate now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN && {
            echo -e "${GREEN}✓ SSL certificate installed and Nginx updated to HTTPS${NC}"
        } || {
            echo -e "${RED}SSL setup failed. You can run it manually later with:${NC}"
            echo "sudo certbot --nginx -d $DOMAIN"
        }
    fi
else
    echo -e "${YELLOW}DNS not configured yet or still propagating${NC}"
    if [ ! -z "$CURRENT_IP" ]; then
        echo "Server IP: $CURRENT_IP"
        echo "Configure DNS A Record: $DOMAIN -> $CURRENT_IP"
    fi
fi

# Final Summary
print_section "14. Setup Complete!"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DEPLOYMENT SUCCESSFUL! 🚀${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Application Directory:${NC} $APP_DIR"
echo -e "${BLUE}API Server Path:${NC} $API_SERVER_PATH"
echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}API Token:${NC} $API_TOKEN"
echo ""
echo -e "${YELLOW}Important Next Steps:${NC}"
echo ""
echo "1. Setup Google Earth Engine authentication:"
echo "   cd $APP_DIR"
echo "   source venv/bin/activate"
echo "   earthengine authenticate"
echo ""
echo "2. Test API locally:"
echo "   curl http://localhost:8000/"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Test API from outside (HTTP - before SSL):"
echo "   curl http://$DOMAIN/"
echo ""
echo "4. After SSL setup, test HTTPS:"
echo "   curl https://$DOMAIN/"
echo ""
echo "4. Monitor logs:"
echo "   sudo journalctl -u hidrologi-api.service -f"
echo "   tail -f $APP_DIR/logs/api.log"
echo ""
echo "5. Check service status:"
echo "   sudo systemctl status hidrologi-api.service"
echo ""
echo -e "${RED}SAVE YOUR API TOKEN SECURELY!${NC}"
echo -e "${RED}Token: $API_TOKEN${NC}"
echo ""
echo "Save tokens to file:"
echo "echo 'SECRET_KEY=$SECRET_KEY' >> $APP_DIR/TOKENS.txt"
echo "echo 'API_TOKEN=$API_TOKEN' >> $APP_DIR/TOKENS.txt"
echo "chmod 600 $APP_DIR/TOKENS.txt"
echo ""
