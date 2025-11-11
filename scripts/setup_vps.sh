#!/bin/bash

#############################################
# HIDROLOGI ML - VPS Setup Script
# Auto-setup untuk deployment di VPS Ubuntu
#############################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_USER="hidrologi"
APP_DIR="/home/$APP_USER/project_hidrologi_ml"
PYTHON_VERSION="3.10"
DOMAIN=""  # Will be set interactively

echo -e "${BLUE}"
echo "================================================"
echo "  HIDROLOGI ML - VPS Deployment Setup"
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

# 1. Gather Information
print_section "1. Gathering Information"

DOMAIN=$(prompt_user "Enter your domain (e.g., api.yourdomain.com): ")
if [ -z "$DOMAIN" ]; then
    echo -e "${RED}Error: Domain is required${NC}"
    exit 1
fi

GIT_REPO=$(prompt_user "Enter your Git repository URL: ")
if [ -z "$GIT_REPO" ]; then
    echo -e "${RED}Error: Git repository URL is required${NC}"
    exit 1
fi

# Generate secure tokens
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo ""
echo -e "${YELLOW}Generated Secure Tokens:${NC}"
echo "SECRET_KEY: $SECRET_KEY"
echo "API_TOKEN: $API_TOKEN"
echo ""
echo -e "${RED}IMPORTANT: Save these tokens securely!${NC}"
read -p "Press Enter to continue..."

# 2. Update System
print_section "2. Updating System"
apt update && apt upgrade -y

# 3. Install Essential Packages
print_section "3. Installing Essential Packages"
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

# 4. Create Application User
print_section "4. Creating Application User"
if id "$APP_USER" &>/dev/null; then
    echo -e "${YELLOW}User $APP_USER already exists${NC}"
else
    useradd -m -s /bin/bash $APP_USER
    echo -e "${GREEN}✓ User $APP_USER created${NC}"
fi

# 5. Clone Repository
print_section "5. Cloning Repository"
if [ -d "$APP_DIR" ]; then
    echo -e "${YELLOW}Directory $APP_DIR already exists${NC}"
    read -p "Delete and re-clone? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $APP_DIR
        sudo -u $APP_USER git clone $GIT_REPO $APP_DIR
    fi
else
    sudo -u $APP_USER git clone $GIT_REPO $APP_DIR
fi

cd $APP_DIR
echo -e "${GREEN}✓ Repository cloned${NC}"

# 6. Setup Python Virtual Environment
print_section "6. Setting up Python Virtual Environment"
sudo -u $APP_USER python3 -m venv $APP_DIR/venv
sudo -u $APP_USER $APP_DIR/venv/bin/pip install --upgrade pip
sudo -u $APP_USER $APP_DIR/venv/bin/pip install -r $APP_DIR/requirements.txt
sudo -u $APP_USER $APP_DIR/venv/bin/pip install gunicorn python-dotenv
echo -e "${GREEN}✓ Python environment setup complete${NC}"

# 7. Create Environment Configuration
print_section "7. Creating Environment Configuration"
cat > $APP_DIR/.env.production <<EOF
# Environment
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
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

chown $APP_USER:$APP_USER $APP_DIR/.env.production
chmod 600 $APP_DIR/.env.production
echo -e "${GREEN}✓ Environment configuration created${NC}"

# 8. Create Required Directories
print_section "8. Creating Required Directories"
sudo -u $APP_USER mkdir -p $APP_DIR/results
sudo -u $APP_USER mkdir -p $APP_DIR/temp
sudo -u $APP_USER mkdir -p $APP_DIR/logs
sudo -u $APP_USER chmod 755 $APP_DIR/results $APP_DIR/temp $APP_DIR/logs
echo -e "${GREEN}✓ Directories created${NC}"

# 9. Setup Systemd Service
print_section "9. Setting up Systemd Service"
cat > /etc/systemd/system/hidrologi-api.service <<EOF
[Unit]
Description=Hidrologi ML API Service
After=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=$APP_DIR/venv/bin/python3 $APP_DIR/project_hidrologi_ml/api_server.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/api.log
StandardError=append:$APP_DIR/logs/api_error.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$APP_DIR/results $APP_DIR/temp $APP_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hidrologi-api.service
echo -e "${GREEN}✓ Systemd service created${NC}"

# 10. Setup Nginx
print_section "10. Setting up Nginx"
mkdir -p /var/www/letsencrypt
chown -R www-data:www-data /var/www/letsencrypt

cat > /etc/nginx/sites-available/hidrologi-api <<EOF
upstream hidrologi_api {
    server 127.0.0.1:8001;
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

# 11. Setup Firewall
print_section "11. Configuring Firewall"
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
echo -e "${GREEN}✓ Firewall configured${NC}"

# 12. Setup Log Rotation
print_section "12. Setting up Log Rotation"
cat > /etc/logrotate.d/hidrologi-api <<EOF
$APP_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    missingok
    create 644 $APP_USER $APP_USER
}
EOF
echo -e "${GREEN}✓ Log rotation configured${NC}"

# 13. Setup Cron for Cleanup
print_section "13. Setting up Automated Cleanup"
(sudo -u $APP_USER crontab -l 2>/dev/null; echo "0 2 * * * find $APP_DIR/results -type d -mtime +30 -exec rm -rf {} + 2>/dev/null") | sudo -u $APP_USER crontab -
echo -e "${GREEN}✓ Cleanup cron job added${NC}"

# 14. Start Service
print_section "14. Starting API Service"
systemctl start hidrologi-api.service
sleep 3
systemctl status hidrologi-api.service --no-pager
echo -e "${GREEN}✓ API service started${NC}"

# 15. Setup SSL Certificate
print_section "15. Setting up SSL Certificate"
echo -e "${YELLOW}Before continuing, make sure DNS A record is pointing to this server!${NC}"
echo "Check with: dig $DOMAIN"
echo ""
read -p "Is DNS configured? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN || {
        echo -e "${RED}SSL setup failed. You can run it manually later with:${NC}"
        echo "sudo certbot --nginx -d $DOMAIN"
    }
    echo -e "${GREEN}✓ SSL certificate installed${NC}"
else
    echo -e "${YELLOW}Skipping SSL setup. Run manually later:${NC}"
    echo "sudo certbot --nginx -d $DOMAIN"
fi

# 16. Final Summary
print_section "16. Setup Complete!"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DEPLOYMENT SUCCESSFUL! 🚀${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}API URL:${NC} https://$DOMAIN"
echo -e "${BLUE}API Token:${NC} $API_TOKEN"
echo ""
echo -e "${YELLOW}Important Next Steps:${NC}"
echo "1. Setup Google Earth Engine authentication:"
echo "   sudo su - $APP_USER"
echo "   source $APP_DIR/venv/bin/activate"
echo "   earthengine authenticate"
echo ""
echo "2. Test API:"
echo "   curl https://$DOMAIN/"
echo ""
echo "3. Monitor logs:"
echo "   sudo journalctl -u hidrologi-api.service -f"
echo "   tail -f $APP_DIR/logs/api.log"
echo ""
echo "4. Check service status:"
echo "   sudo systemctl status hidrologi-api.service"
echo ""
echo -e "${RED}SAVE YOUR API TOKEN SECURELY!${NC}"
echo -e "${RED}Token: $API_TOKEN${NC}"
echo ""
echo "Full documentation: $APP_DIR/DEPLOYMENT_GUIDE.md"
echo ""
