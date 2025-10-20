# 🚀 Deployment Guide - Hidrologi ML API to VPS

## 📋 Prerequisites

- VPS dengan Ubuntu 20.04/22.04 (atau Debian-based)
- Root atau sudo access
- Domain: itriverdna.my.id
- GitHub repository dengan code

---

## 🔧 Step 1: Persiapan VPS & Install Dependencies

### 1.1 Login ke VPS
```bash
ssh root@your-vps-ip
# atau
ssh your-username@your-vps-ip
```

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.3 Install Python 3.11+ & Dependencies
```bash
# Install Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install Git
sudo apt install git -y

# Install build tools untuk compile packages
sudo apt install build-essential libssl-dev libffi-dev -y
sudo apt install libgdal-dev gdal-bin -y
sudo apt install libspatialindex-dev -y

# Install supervisor untuk process management
sudo apt install supervisor -y

# Install nginx (optional, untuk reverse proxy)
sudo apt install nginx -y
```

---

## 📦 Step 2: Clone Repository & Setup Project

### 2.1 Navigate ke Directory
```bash
cd /var/www/itriverdna.my.id/public_html
```

### 2.2 Clone Repository
```bash
# Hapus folder lama jika ada
sudo rm -rf project_hidrologi_ml

# Clone dari GitHub
sudo git clone https://github.com/YOUR_USERNAME/project_hidrologi_ml.git
cd project_hidrologi_ml
```

### 2.3 Set Permissions
```bash
# Ubah ownership ke user www-data (nginx/apache user)
sudo chown -R www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml

# Set proper permissions
sudo chmod -R 755 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml

# Buat folder results jika belum ada
sudo mkdir -p /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
sudo chown -R www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
sudo chmod -R 775 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
```

---

## 🐍 Step 3: Setup Virtual Environment

### 3.1 Create Virtual Environment
```bash
cd /var/www/itriverdna.my.id/public_html/project_hidrologi_ml

# Create venv dengan Python 3.11
sudo -u www-data python3.11 -m venv venv

# Activate venv
sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade pip setuptools wheel"
```

### 3.2 Install Python Packages
```bash
# Install semua dependencies
sudo -u www-data bash -c "source venv/bin/activate && pip install -r requirements.txt"

# Jika ada error dengan GDAL, install manual:
sudo -u www-data bash -c "source venv/bin/activate && pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option='-I/usr/include/gdal'"
```

---

## 🔐 Step 4: Setup Environment Variables

### 4.1 Create .env.production File
```bash
sudo nano /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/.env.production
```

### 4.2 Add Configuration
```env
# Environment
ENVIRONMENT=production

# Server Configuration
HOST=0.0.0.0
PORT=8080
API_BASE_URL=https://itriverdna.my.id/api

# Google Earth Engine
GEE_PROJECT_ID=your-gee-project-id
GEE_SERVICE_ACCOUNT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json

# Storage Paths
RESULTS_DIR=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
JOBS_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/jobs.json

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/api_server.log

# Security
ALLOWED_ORIGINS=https://itriverdna.my.id,http://localhost
MAX_JOBS_PER_HOUR=10

# TensorFlow
TF_CPP_MIN_LOG_LEVEL=1
```

Save dengan `Ctrl+O`, Enter, `Ctrl+X`

### 4.3 Upload GEE Credentials
```bash
# Copy your GEE service account JSON ke server
# Dari local machine:
scp /path/to/your/gee-credentials.json root@your-vps:/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/

# Di server, set permissions:
sudo chown www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
sudo chmod 600 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
```

---

## 🔄 Step 5: Setup Supervisor (Process Manager untuk 24/7 Uptime)

### 5.1 Create Supervisor Config
```bash
sudo nano /etc/supervisor/conf.d/hidrologi-api.conf
```

### 5.2 Add Configuration
```ini
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

[program:hidrologi-api-worker]
directory=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml
command=/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/venv/bin/python /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/project_hidrologi_ml/api_server.py
user=www-data
process_name=%(program_name)s_%(process_num)02d
numprocs=2
autostart=true
autorestart=true
startretries=10
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/supervisor/hidrologi-api-worker-%(process_num)02d.err.log
stdout_logfile=/var/log/supervisor/hidrologi-api-worker-%(process_num)02d.out.log
environment=PATH="/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/venv/bin"
```

Save dengan `Ctrl+O`, Enter, `Ctrl+X`

### 5.3 Reload & Start Supervisor
```bash
# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start the service
sudo supervisorctl start hidrologi-api:*

# Check status
sudo supervisorctl status

# Expected output:
# hidrologi-api                    RUNNING   pid 12345, uptime 0:00:05
# hidrologi-api-worker:hidrologi-api-worker_00   RUNNING   pid 12346, uptime 0:00:05
# hidrologi-api-worker:hidrologi-api-worker_01   RUNNING   pid 12347, uptime 0:00:05
```

---

## 🌐 Step 6: Setup Nginx Reverse Proxy (Optional tapi Recommended)

### 6.1 Create Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/hidrologi-api
```

### 6.2 Add Configuration
```nginx
upstream hidrologi_backend {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 80;
    server_name itriverdna.my.id api.itriverdna.my.id;

    # Increase client body size for file uploads
    client_max_body_size 100M;

    # API endpoints
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
        
        # Timeout settings untuk long-running requests
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }

    # Serve static files directly
    location /api/files/ {
        alias /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/;
        autoindex off;
        add_header Cache-Control "public, max-age=3600";
    }

    # Health check endpoint
    location /api/health {
        proxy_pass http://hidrologi_backend/health;
        access_log off;
    }

    # Logs
    access_log /var/log/nginx/hidrologi-api-access.log;
    error_log /var/log/nginx/hidrologi-api-error.log;
}
```

Save dengan `Ctrl+O`, Enter, `Ctrl+X`

### 6.3 Enable Site & Reload Nginx
```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

# Enable nginx to start on boot
sudo systemctl enable nginx
```

### 6.4 Setup SSL dengan Let's Encrypt (HTTPS)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d itriverdna.my.id -d api.itriverdna.my.id

# Follow prompts:
# - Enter email
# - Agree to terms
# - Choose to redirect HTTP to HTTPS (recommended)

# Auto-renewal sudah setup automatically
# Test renewal:
sudo certbot renew --dry-run
```

---

## 🔍 Step 7: Monitoring & Maintenance

### 7.1 Check API Status
```bash
# Via supervisor
sudo supervisorctl status hidrologi-api:*

# Check logs
sudo tail -f /var/log/supervisor/hidrologi-api.out.log
sudo tail -f /var/log/supervisor/hidrologi-api.err.log

# Check nginx logs
sudo tail -f /var/log/nginx/hidrologi-api-access.log
sudo tail -f /var/log/nginx/hidrologi-api-error.log
```

### 7.2 Restart Services
```bash
# Restart API
sudo supervisorctl restart hidrologi-api:*

# Reload nginx
sudo systemctl reload nginx

# Restart all services
sudo supervisorctl restart all
sudo systemctl restart nginx
```

### 7.3 Update Code dari GitHub
```bash
cd /var/www/itriverdna.my.id/public_html/project_hidrologi_ml

# Pull latest code
sudo -u www-data git pull origin main

# Install/update dependencies if needed
sudo -u www-data bash -c "source venv/bin/activate && pip install -r requirements.txt"

# Restart API
sudo supervisorctl restart hidrologi-api:*
```

### 7.4 Disk Space Management
```bash
# Check disk usage
df -h

# Clean old results (older than 30 days)
find /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/* -mtime +30 -type d -exec rm -rf {} +

# Setup cron job untuk auto-cleanup
sudo crontab -e

# Add this line:
0 2 * * * find /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/* -mtime +30 -type d -exec rm -rf {} +
```

---

## 🛠️ Step 8: Testing

### 8.1 Test dari Browser
```
https://itriverdna.my.id/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-20T12:00:00",
  "uptime": 3600
}
```

### 8.2 Test API Endpoint
```bash
# From local machine or VPS
curl -X POST https://itriverdna.my.id/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": 110.4203,
    "latitude": -7.7956,
    "start": "2023-01-01",
    "end": "2023-12-31"
  }'
```

### 8.3 Test dari Laravel Application
Update `.env` di Laravel:
```env
HIDROLOGI_API_URL=https://itriverdna.my.id/api
```

---

## 📊 Step 9: Performance Optimization

### 9.1 Setup Swap (jika RAM terbatas)
```bash
# Create 4GB swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 9.2 Setup Firewall
```bash
# Install UFW
sudo apt install ufw -y

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow API port (jika direct access)
sudo ufw allow 8080/tcp

# Enable firewall
sudo ufw enable
```

### 9.3 Setup Log Rotation
```bash
sudo nano /etc/logrotate.d/hidrologi-api
```

Add:
```
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

/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/api_server.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
}
```

---

## 🔧 Troubleshooting

### Problem 1: API tidak start
```bash
# Check logs
sudo supervisorctl tail -f hidrologi-api stderr

# Common issues:
# - Port already in use: Change PORT in .env.production
# - Permission denied: Check file permissions
# - Module not found: Reinstall requirements
```

### Problem 2: GEE Authentication Failed
```bash
# Verify credentials file exists
ls -la /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json

# Check permissions
sudo chmod 600 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
sudo chown www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json

# Verify GEE project ID in .env.production
```

### Problem 3: High Memory Usage
```bash
# Check memory
free -h

# Restart services
sudo supervisorctl restart hidrologi-api:*

# Reduce number of workers in supervisor config
```

### Problem 4: Slow Response
```bash
# Check if running
sudo supervisorctl status

# Check system load
top
htop

# Check nginx status
sudo systemctl status nginx
```

---

## 📝 Useful Commands

### Supervisor Commands
```bash
sudo supervisorctl status                    # Check status
sudo supervisorctl start hidrologi-api:*     # Start service
sudo supervisorctl stop hidrologi-api:*      # Stop service
sudo supervisorctl restart hidrologi-api:*   # Restart service
sudo supervisorctl tail hidrologi-api stdout # View logs
sudo supervisorctl tail -f hidrologi-api     # Follow logs
```

### Nginx Commands
```bash
sudo systemctl status nginx      # Check status
sudo systemctl start nginx       # Start nginx
sudo systemctl stop nginx        # Stop nginx
sudo systemctl restart nginx     # Restart nginx
sudo systemctl reload nginx      # Reload config
sudo nginx -t                    # Test config
```

### System Monitoring
```bash
htop                            # Interactive process viewer
df -h                           # Disk usage
free -h                         # Memory usage
netstat -tulpn | grep 8080      # Check port usage
journalctl -u supervisor -f     # Supervisor logs
```

---

## 🎯 Summary

**Setelah semua langkah di atas, API akan:**
- ✅ Running 24/7 dengan auto-restart
- ✅ Accessible via HTTPS (SSL)
- ✅ Behind Nginx reverse proxy untuk performance
- ✅ Auto-update dengan git pull
- ✅ Managed logs dengan rotation
- ✅ Firewall protection
- ✅ Auto-cleanup old results

**API Endpoints:**
- Production: `https://itriverdna.my.id/api/`
- Health Check: `https://itriverdna.my.id/api/health`
- Generate Job: `https://itriverdna.my.id/api/generate`

**Management:**
- Logs: `/var/log/supervisor/`
- Results: `/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/`
- Config: `/etc/supervisor/conf.d/hidrologi-api.conf`

---

## 📞 Support

Jika ada masalah, check:
1. Supervisor logs: `sudo supervisorctl tail -f hidrologi-api`
2. Nginx logs: `sudo tail -f /var/log/nginx/hidrologi-api-error.log`
3. System logs: `journalctl -xe`

**Good luck! 🚀**
