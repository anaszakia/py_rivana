# 🚀 QUICK START - Deploy ke VPS

## Cara Tercepat Deploy

### 1. **Dari Local Machine** - Upload ke VPS
```bash
# Clone/download project ini dulu
git clone https://github.com/YOUR_USERNAME/project_hidrologi_ml.git
cd project_hidrologi_ml

# Upload ke VPS menggunakan SCP
scp -r . root@YOUR_VPS_IP:/root/project_hidrologi_ml
```

### 2. **Di VPS** - Jalankan Setup Script
```bash
# Login ke VPS
ssh root@YOUR_VPS_IP

# Navigate ke directory
cd /root/project_hidrologi_ml

# Beri permission execute
chmod +x scripts/*.sh

# Jalankan auto-setup script
sudo bash scripts/setup_vps.sh
```

Script akan menanyakan:
- **Domain**: `api.yourdomain.com` (pastikan DNS sudah pointing!)
- **Git Repository**: URL repository Anda

Script akan otomatis:
- ✅ Install semua dependencies
- ✅ Setup user & permissions
- ✅ Clone repository
- ✅ Setup Python virtual environment
- ✅ Generate secure tokens
- ✅ Configure Nginx
- ✅ Setup SSL certificate
- ✅ Start service

### 3. **Setup Earth Engine**
```bash
# Switch ke user aplikasi
sudo su - hidrologi

# Activate virtual environment
cd project_hidrologi_ml
source venv/bin/activate

# Authenticate Earth Engine
earthengine authenticate
```

### 4. **Test API**
```bash
# Test dari VPS
curl https://api.yourdomain.com/

# Test dari local machine
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     https://api.yourdomain.com/jobs
```

---

## Scripts Yang Tersedia

### 1. **setup_vps.sh** - Initial Setup
```bash
sudo bash scripts/setup_vps.sh
```
Untuk deployment awal ke VPS baru.

### 2. **update_app.sh** - Update Code
```bash
bash scripts/update_app.sh
```
Untuk update code dari Git (setelah ada perubahan).

### 3. **monitor.sh** - Real-time Monitoring
```bash
bash scripts/monitor.sh
```
Dashboard monitoring real-time untuk:
- Service status
- CPU/Memory usage
- Active jobs
- Recent errors

---

## Struktur Files Deployment

```
project_hidrologi_ml/
├── DEPLOYMENT_GUIDE.md          # Dokumentasi lengkap
├── QUICK_START.md               # This file
├── .env.production.example      # Template environment
├── scripts/
│   ├── setup_vps.sh            # Auto-setup VPS
│   ├── update_app.sh           # Update aplikasi
│   └── monitor.sh              # Monitoring dashboard
├── configs/
│   ├── nginx.conf              # Nginx configuration
│   ├── hidrologi-api.service   # Systemd service
│   └── logrotate.conf          # Log rotation
└── project_hidrologi_ml/
    ├── api_server.py           # Main API server
    ├── config.py               # Configuration manager
    └── ...
```

---

## Common Commands

### Service Management
```bash
# Start service
sudo systemctl start hidrologi-api

# Stop service
sudo systemctl stop hidrologi-api

# Restart service
sudo systemctl restart hidrologi-api

# Check status
sudo systemctl status hidrologi-api

# View logs
sudo journalctl -u hidrologi-api -f
```

### Nginx Management
```bash
# Test configuration
sudo nginx -t

# Reload configuration
sudo systemctl reload nginx

# View logs
sudo tail -f /var/log/nginx/hidrologi_access.log
sudo tail -f /var/log/nginx/hidrologi_error.log
```

### Monitoring
```bash
# Real-time monitoring dashboard
bash scripts/monitor.sh

# Check disk usage
df -h

# Check memory
free -h

# Check processes
htop
```

---

## Troubleshooting Cepat

### Service tidak start?
```bash
# Check logs
sudo journalctl -u hidrologi-api -n 50

# Check file permissions
ls -la /home/hidrologi/project_hidrologi_ml/
```

### API tidak bisa diakses?
```bash
# Check service
sudo systemctl status hidrologi-api

# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# Check firewall
sudo ufw status
```

### Earth Engine error?
```bash
# Re-authenticate
sudo su - hidrologi
cd project_hidrologi_ml
source venv/bin/activate
earthengine authenticate
```

---

## 🎯 Checklist Deployment

Pastikan ini sudah done:

- [ ] VPS ready (Ubuntu 20.04/22.04, min 4GB RAM)
- [ ] Domain sudah pointing ke VPS IP
- [ ] Script `setup_vps.sh` berhasil dijalankan
- [ ] Earth Engine ter-authenticate
- [ ] SSL certificate ter-install
- [ ] API bisa diakses dari external: `https://api.yourdomain.com`
- [ ] Test generate job berhasil
- [ ] Service otomatis start saat reboot: `sudo systemctl enable hidrologi-api`

---

## 📞 Need Help?

Lihat dokumentasi lengkap: **DEPLOYMENT_GUIDE.md**

Atau check logs:
```bash
sudo journalctl -u hidrologi-api -f
tail -f ~/project_hidrologi_ml/logs/api.log
```

---

**Happy Deploying! 🚀**
