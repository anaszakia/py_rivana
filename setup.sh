#!/bin/bash

# ============================================
# Quick Setup Script - Hidrologi ML API
# Run this ONCE after cloning to VPS
# ============================================

set -e

echo "🔧 Quick Setup - Hidrologi ML API"
echo "=================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root: sudo bash setup.sh"
    exit 1
fi

# Variables
PROJECT_DIR="/var/www/itriverdna.my.id/public_html/project_hidrologi_ml"
DOMAIN="itriverdna.my.id"

# Step 1: Install dependencies
print_step "1/8 Installing system dependencies..."
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev \
    git build-essential libssl-dev libffi-dev \
    libgdal-dev gdal-bin libspatialindex-dev \
    supervisor nginx certbot python3-certbot-nginx
print_message "✓ Dependencies installed"

# Step 2: Setup project directory
print_step "2/8 Setting up project directory..."
cd "$PROJECT_DIR"
chown -R www-data:www-data "$PROJECT_DIR"
chmod -R 755 "$PROJECT_DIR"
mkdir -p results
chmod -R 775 results
print_message "✓ Directory setup complete"

# Step 3: Create virtual environment
print_step "3/8 Creating Python virtual environment..."
sudo -u www-data python3.11 -m venv venv
sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade pip setuptools wheel"
print_message "✓ Virtual environment created"

# Step 4: Install Python packages
print_step "4/8 Installing Python packages (this may take a while)..."
sudo -u www-data bash -c "source venv/bin/activate && pip install -r requirements.txt"
print_message "✓ Python packages installed"

# Step 5: Setup environment file
print_step "5/8 Creating environment configuration..."
if [ ! -f "$PROJECT_DIR/.env.production" ]; then
    cat > "$PROJECT_DIR/.env.production" << 'EOF'
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8080
API_BASE_URL=https://itriverdna.my.id/api
GEE_PROJECT_ID=your-gee-project-id
GEE_SERVICE_ACCOUNT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
RESULTS_DIR=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
JOBS_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/jobs.json
LOG_LEVEL=INFO
LOG_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/api_server.log
TF_CPP_MIN_LOG_LEVEL=1
EOF
    chown www-data:www-data "$PROJECT_DIR/.env.production"
    print_warning "⚠️  Please edit .env.production and add your GEE credentials!"
    print_message "✓ Environment file created"
else
    print_message "✓ Environment file already exists"
fi

# Step 6: Setup Supervisor
print_step "6/8 Configuring Supervisor..."
cat > /etc/supervisor/conf.d/hidrologi-api.conf << 'EOF'
[program:hidrologi-api]
directory=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml
command=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/venv/bin/python /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/project_hidrologi_ml/api_server.py
user=www-data
autostart=true
autorestart=true
startretries=10
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/supervisor/hidrologi-api.err.log
stdout_logfile=/var/log/supervisor/hidrologi-api.out.log
environment=PATH="/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/venv/bin"
EOF

supervisorctl reread
supervisorctl update
print_message "✓ Supervisor configured"

# Step 7: Setup Nginx
print_step "7/8 Configuring Nginx..."
cat > /etc/nginx/sites-available/hidrologi-api << 'EOF'
upstream hidrologi_backend {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name itriverdna.my.id api.itriverdna.my.id;
    client_max_body_size 100M;

    location /api/ {
        proxy_pass http://hidrologi_backend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }

    location /api/files/ {
        alias /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/;
        autoindex off;
        add_header Cache-Control "public, max-age=3600";
    }

    access_log /var/log/nginx/hidrologi-api-access.log;
    error_log /var/log/nginx/hidrologi-api-error.log;
}
EOF

ln -sf /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
systemctl enable nginx
print_message "✓ Nginx configured"

# Step 8: Setup Firewall
print_step "8/8 Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw allow ssh
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable
    print_message "✓ Firewall configured"
else
    print_warning "UFW not installed, skipping firewall setup"
fi

# Setup log rotation
cat > /etc/logrotate.d/hidrologi-api << 'EOF'
/var/log/supervisor/hidrologi-api*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        supervisorctl restart hidrologi-api:*
    endscript
}
EOF

# Setup auto-cleanup cron
(crontab -l 2>/dev/null; echo "0 2 * * * find /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/* -mtime +30 -type d -exec rm -rf {} +") | crontab -

echo ""
print_message "=================================="
print_message "✅ Setup completed successfully!"
print_message "=================================="
echo ""
print_warning "NEXT STEPS:"
echo "1. Upload your GEE credentials JSON file to:"
echo "   $PROJECT_DIR/gee-credentials.json"
echo ""
echo "2. Edit environment file:"
echo "   sudo nano $PROJECT_DIR/.env.production"
echo ""
echo "3. Start the service:"
echo "   sudo supervisorctl start hidrologi-api:*"
echo ""
echo "4. Setup SSL certificate:"
echo "   sudo certbot --nginx -d itriverdna.my.id -d api.itriverdna.my.id"
echo ""
echo "5. Check status:"
echo "   sudo supervisorctl status"
echo "   curl http://localhost:8080/health"
echo ""
print_message "For detailed documentation, see DEPLOYMENT_GUIDE.md"
echo ""
