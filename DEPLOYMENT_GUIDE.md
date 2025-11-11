# 🚀 PANDUAN DEPLOYMENT PROJECT HIDROLOGI ML KE VPS

## 📋 Daftar Isi
1. [Persiapan VPS](#persiapan-vps)
2. [Instalasi Dependencies](#instalasi-dependencies)
3. [Clone & Setup Project](#clone--setup-project)
4. [Konfigurasi Environment](#konfigurasi-environment)
5. [Setup Google Earth Engine](#setup-google-earth-engine)
6. [Setup Systemd Service](#setup-systemd-service)
7. [Setup Nginx (Reverse Proxy)](#setup-nginx-reverse-proxy)
8. [SSL Certificate dengan Let's Encrypt](#ssl-certificate-dengan-lets-encrypt)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

---

## 🖥️ Persiapan VPS

### Spesifikasi Minimal VPS:
- **CPU**: 2 vCPU atau lebih
- **RAM**: 4 GB (8 GB recommended untuk TensorFlow)
- **Storage**: 20 GB SSD
- **OS**: Ubuntu 20.04 LTS / 22.04 LTS
- **Port**: 80, 443, 8001 (untuk API)

### 1. Login ke VPS
```bash
ssh root@YOUR_VPS_IP
# atau jika menggunakan user biasa:
ssh your_username@YOUR_VPS_IP
```

### 2. Update Sistem
```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Essentials
```bash
sudo apt install -y build-essential git curl wget vim nano htop \
    python3-dev python3-pip python3-venv pkg-config libhdf5-dev
```

---

## 📦 Instalasi Dependencies

### 1. Install Python 3.9+ (Jika belum ada)
```bash
# Check versi Python
python3 --version

# Jika < 3.9, install Python 3.10
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### 2. Install GDAL & Geospatial Libraries
```bash
sudo apt install -y gdal-bin libgdal-dev libspatialindex-dev
```

### 3. Install Nginx
```bash
sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

---

## 📥 Clone & Setup Project

### 1. Buat User untuk Aplikasi (Security Best Practice)
```bash
sudo useradd -m -s /bin/bash hidrologi
sudo passwd hidrologi  # Set password
```

### 2. Switch ke User Aplikasi
```bash
sudo su - hidrologi
```

### 3. Clone Repository
```bash
# GANTI dengan URL repository Anda
cd ~
git clone https://github.com/YOUR_USERNAME/project_hidrologi_ml.git
cd project_hidrologi_ml
```

### 4. Buat Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies Python
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional dependencies untuk production
pip install gunicorn python-dotenv
```

---

## ⚙️ Konfigurasi Environment

### 1. Buat File Environment Production
```bash
nano .env.production
```

### 2. Isi dengan Konfigurasi Berikut:
```ini
# Environment
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
DEBUG=False

# Paths (absolute paths di VPS)
RESULTS_DIR=/home/hidrologi/project_hidrologi_ml/results
TEMP_DIR=/home/hidrologi/project_hidrologi_ml/temp

# Earth Engine
EE_AUTHENTICATED=True

# Security - GANTI dengan token aman Anda!
SECRET_KEY=YOUR_SECURE_SECRET_KEY_CHANGE_THIS
API_TOKEN=YOUR_SECURE_API_TOKEN_CHANGE_THIS

# Limits
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT=1800

# Logging
LOG_LEVEL=INFO
LOG_FILE=/home/hidrologi/project_hidrologi_ml/logs/api.log

# Performance
ENABLE_CORS=True
ENABLE_CACHE=True
CACHE_TTL=600

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### 3. Generate Secure Tokens
```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"

# Generate API_TOKEN
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Simpan tokens ini dan update di `.env.production`

### 4. Buat Direktori yang Diperlukan
```bash
mkdir -p results temp logs
chmod 755 results temp logs
```

---

## 🌍 Setup Google Earth Engine

### 1. Install Earth Engine API
```bash
# Sudah ter-install dari requirements.txt
pip show earthengine-api
```

### 2. Autentikasi Earth Engine
```bash
earthengine authenticate
```

Ikuti petunjuk:
1. Buka URL yang diberikan di browser
2. Login dengan akun Google yang sudah registered di Earth Engine
3. Copy authentication code
4. Paste di terminal

### 3. Upload Credentials (Jika sudah punya)
Jika Anda sudah punya `gee-credentials.json`:
```bash
# Copy dari local ke VPS
# Di local machine:
scp gee-credentials.json hidrologi@YOUR_VPS_IP:~/project_hidrologi_ml/project_hidrologi_ml/

# Di VPS, set permissions:
chmod 600 ~/project_hidrologi_ml/project_hidrologi_ml/gee-credentials.json
```

---

## 🔧 Setup Systemd Service

### 1. Buat Systemd Service File
```bash
sudo nano /etc/systemd/system/hidrologi-api.service
```

### 2. Isi dengan Konfigurasi Berikut:
```ini
[Unit]
Description=Hidrologi ML API Service
After=network.target

[Service]
Type=simple
User=hidrologi
Group=hidrologi
WorkingDirectory=/home/hidrologi/project_hidrologi_ml
Environment="PATH=/home/hidrologi/project_hidrologi_ml/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=/home/hidrologi/project_hidrologi_ml/venv/bin/python3 /home/hidrologi/project_hidrologi_ml/project_hidrologi_ml/api_server.py
Restart=always
RestartSec=10
StandardOutput=append:/home/hidrologi/project_hidrologi_ml/logs/api.log
StandardError=append:/home/hidrologi/project_hidrologi_ml/logs/api_error.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/hidrologi/project_hidrologi_ml/results /home/hidrologi/project_hidrologi_ml/temp /home/hidrologi/project_hidrologi_ml/logs

[Install]
WantedBy=multi-user.target
```

### 3. Enable & Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable hidrologi-api.service
sudo systemctl start hidrologi-api.service
```

### 4. Check Status
```bash
sudo systemctl status hidrologi-api.service
```

### 5. View Logs
```bash
# Tail logs real-time
sudo journalctl -u hidrologi-api.service -f

# View all logs
sudo journalctl -u hidrologi-api.service

# Last 100 lines
sudo journalctl -u hidrologi-api.service -n 100
```

---

## 🌐 Setup Nginx (Reverse Proxy)

### 1. Buat Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/hidrologi-api
```

### 2. Isi dengan Konfigurasi Berikut:
```nginx
# Upstream untuk API server
upstream hidrologi_api {
    server 127.0.0.1:8001;
    keepalive 32;
}

# HTTP Server - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name api.yourdomain.com;  # GANTI dengan domain Anda
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }
    
    # Redirect all HTTP to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS Server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.yourdomain.com;  # GANTI dengan domain Anda
    
    # SSL Configuration (akan disetup oleh certbot)
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Max upload size (untuk large data)
    client_max_body_size 100M;
    
    # Timeouts (karena ML process bisa lama)
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
    send_timeout 600s;
    
    # Logs
    access_log /var/log/nginx/hidrologi_access.log;
    error_log /var/log/nginx/hidrologi_error.log;
    
    # API Proxy
    location / {
        proxy_pass http://hidrologi_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # Static files (untuk download hasil)
    location /download/ {
        alias /home/hidrologi/project_hidrologi_ml/results/;
        autoindex off;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

### 3. Enable Site
```bash
# Test konfigurasi
sudo nginx -t

# Enable site
sudo ln -s /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/

# Reload Nginx
sudo systemctl reload nginx
```

---

## 🔐 SSL Certificate dengan Let's Encrypt

### 1. Install Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 2. Buat Directory untuk Challenge
```bash
sudo mkdir -p /var/www/letsencrypt
sudo chown -R www-data:www-data /var/www/letsencrypt
```

### 3. Pastikan Domain Sudah Pointing ke VPS
Sebelum lanjut, pastikan DNS record Anda sudah pointing:
```
A Record: api.yourdomain.com -> YOUR_VPS_IP
```

Check dengan:
```bash
dig api.yourdomain.com
# atau
nslookup api.yourdomain.com
```

### 4. Generate SSL Certificate
```bash
sudo certbot --nginx -d api.yourdomain.com
```

Ikuti petunjuk:
- Email: masukkan email Anda
- Terms: setuju
- Redirect: pilih Yes untuk auto-redirect HTTP ke HTTPS

### 5. Test Auto-Renewal
```bash
sudo certbot renew --dry-run
```

Certificate akan auto-renew otomatis setiap 3 bulan.

---

## 📊 Monitoring & Maintenance

### 1. Setup Log Rotation
```bash
sudo nano /etc/logrotate.d/hidrologi-api
```

Isi:
```
/home/hidrologi/project_hidrologi_ml/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    missingok
    create 644 hidrologi hidrologi
}
```

### 2. Cron Job untuk Cleanup Old Results
```bash
crontab -e
```

Tambahkan:
```cron
# Cleanup results older than 30 days, setiap hari jam 2 pagi
0 2 * * * find /home/hidrologi/project_hidrologi_ml/results -type d -mtime +30 -exec rm -rf {} + 2>/dev/null
```

### 3. Monitoring Commands
```bash
# Check service status
sudo systemctl status hidrologi-api

# Tail logs
tail -f ~/project_hidrologi_ml/logs/api.log

# Check memory usage
free -h

# Check disk usage
df -h

# Check CPU & RAM
htop

# Check open ports
sudo netstat -tulpn | grep :8001

# Check Nginx logs
sudo tail -f /var/log/nginx/hidrologi_access.log
sudo tail -f /var/log/nginx/hidrologi_error.log
```

### 4. Restart Service (Jika Ada Update)
```bash
# Pull latest code
cd ~/project_hidrologi_ml
git pull origin main

# Activate venv & update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart hidrologi-api.service

# Check status
sudo systemctl status hidrologi-api.service
```

---

## 🔥 Setup Firewall (UFW)

```bash
# Enable UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Check status
sudo ufw status
```

---

## 🧪 Testing Deployment

### 1. Test API dari VPS
```bash
# Test health check
curl http://localhost:8001/

# Test dengan Bearer Token
curl -H "Authorization: Bearer YOUR_API_TOKEN" http://localhost:8001/jobs
```

### 2. Test dari External (Local Machine)
```bash
# Test HTTPS
curl https://api.yourdomain.com/

# Test dengan authentication
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     -H "Content-Type: application/json" \
     https://api.yourdomain.com/jobs
```

### 3. Test Generate Job
```bash
curl -X POST https://api.yourdomain.com/generate \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": 110.3695,
    "latitude": -7.7956,
    "start": "2023-01-01",
    "end": "2023-12-31"
  }'
```

---

## 🐛 Troubleshooting

### Service Tidak Start

```bash
# Check logs detail
sudo journalctl -u hidrologi-api.service -n 50 --no-pager

# Check Python errors
cat ~/project_hidrologi_ml/logs/api_error.log

# Check file permissions
ls -la ~/project_hidrologi_ml/project_hidrologi_ml/api_server.py
```

### Port 8001 Sudah Digunakan
```bash
# Check process menggunakan port 8001
sudo lsof -i :8001

# Kill process jika perlu
sudo kill -9 PID
```

### Nginx Error
```bash
# Test konfigurasi
sudo nginx -t

# Check error logs
sudo tail -f /var/log/nginx/error.log

# Reload Nginx
sudo systemctl reload nginx
```

### Earth Engine Authentication Error
```bash
# Re-authenticate
earthengine authenticate

# Check credentials
ls -la ~/.config/earthengine/credentials
```

### Memory Issues (TensorFlow)
```bash
# Check memory
free -h

# Jika RAM kurang, add swap:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 🚀 Production Best Practices

### 1. Backup Regular
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/home/hidrologi/backups"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf $BACKUP_DIR/project_backup_$DATE.tar.gz ~/project_hidrologi_ml/
find $BACKUP_DIR -name "project_backup_*.tar.gz" -mtime +7 -delete
```

### 2. Monitoring dengan Tools
- Install **htop**: `sudo apt install htop`
- Install **Prometheus** & **Grafana** untuk monitoring advanced
- Setup **Email Alerts** untuk service down

### 3. Security Hardening
- Disable root login: `sudo nano /etc/ssh/sshd_config`
- Use SSH keys instead of passwords
- Keep system updated: `sudo apt update && sudo apt upgrade`
- Regular security audits

---

## 📝 Dokumentasi API untuk Client

Base URL Production: `https://api.yourdomain.com`

### Authentication
Semua request harus include Bearer Token di header:
```
Authorization: Bearer YOUR_API_TOKEN
```

### Endpoints:
- `POST /generate` - Generate new analysis job
- `GET /status/{job_id}` - Check job status
- `GET /result/{job_id}` - Get job results
- `GET /files/{job_id}` - List all files
- `GET /download/{job_id}/{filename}` - Download file
- `GET /jobs` - List all jobs

Untuk dokumentasi lengkap, lihat: `API_DOCUMENTATION.md`

---

## ✅ Checklist Deployment

- [ ] VPS ready (Ubuntu 22.04, 4GB RAM)
- [ ] User aplikasi created
- [ ] Repository cloned
- [ ] Virtual environment setup
- [ ] Dependencies installed
- [ ] `.env.production` configured dengan secure tokens
- [ ] Earth Engine authenticated
- [ ] Systemd service running
- [ ] Nginx configured
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Logs rotation setup
- [ ] Backup script setup
- [ ] API tested dari external

---

## 📞 Support

Jika ada masalah saat deployment, check:
1. Service logs: `sudo journalctl -u hidrologi-api.service`
2. Application logs: `~/project_hidrologi_ml/logs/api.log`
3. Nginx logs: `/var/log/nginx/hidrologi_error.log`

**Good luck dengan deployment! 🚀**
