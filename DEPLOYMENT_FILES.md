# 📦 Files Deployment yang Sudah Dibuat

Berikut adalah daftar lengkap file untuk deployment project Hidrologi ML ke VPS:

## 📄 Dokumentasi

### 1. **DEPLOYMENT_GUIDE.md** ⭐
Dokumentasi lengkap step-by-step untuk deployment ke VPS, termasuk:
- Persiapan VPS
- Instalasi dependencies
- Setup Google Earth Engine
- Konfigurasi Nginx & SSL
- Monitoring & maintenance
- Troubleshooting lengkap

### 2. **QUICK_START.md** 🚀
Panduan cepat untuk deployment dalam 4 langkah:
- Upload project ke VPS
- Jalankan auto-setup script
- Setup Earth Engine
- Test API

---

## 🔧 Scripts Automation

### 1. **scripts/setup_vps.sh** 
Auto-setup script untuk deployment awal. Melakukan:
- Install semua dependencies (Python, GDAL, Nginx, Certbot)
- Buat user aplikasi
- Clone repository
- Setup virtual environment & install packages
- Generate secure tokens otomatis
- Configure Nginx reverse proxy
- Setup Systemd service
- Install SSL certificate
- Configure firewall

**Cara pakai:**
```bash
sudo bash scripts/setup_vps.sh
```

### 2. **scripts/update_app.sh**
Script untuk update aplikasi setelah ada perubahan code:
- Backup otomatis sebelum update
- Pull latest code dari Git
- Update dependencies
- Restart service
- Test API

**Cara pakai:**
```bash
bash scripts/update_app.sh
```

### 3. **scripts/monitor.sh**
Real-time monitoring dashboard yang menampilkan:
- Service status
- CPU & memory usage
- Disk usage
- Active jobs
- Recent errors
- Quick actions (restart, view logs, dll)

**Cara pakai:**
```bash
bash scripts/monitor.sh
```

### 4. **scripts/backup.sh**
Automated backup script untuk:
- Backup application code
- Backup configuration
- Backup recent results (7 hari terakhir)
- Auto-cleanup old backups

**Cara pakai:**
```bash
bash scripts/backup.sh
```

**Setup cron untuk auto-backup harian:**
```bash
crontab -e
# Add: 0 3 * * * /home/hidrologi/project_hidrologi_ml/scripts/backup.sh
```

---

## ⚙️ File Konfigurasi

### 1. **configs/nginx.conf**
Nginx configuration untuk reverse proxy:
- HTTP to HTTPS redirect
- SSL/TLS configuration
- Security headers
- Long timeout untuk ML processing
- Static file serving
- Rate limiting ready

**Install:**
```bash
sudo cp configs/nginx.conf /etc/nginx/sites-available/hidrologi-api
sudo ln -s /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 2. **configs/hidrologi-api.service**
Systemd service configuration:
- Auto-restart on failure
- Logging to files
- Security hardening
- Resource limits

**Install:**
```bash
sudo cp configs/hidrologi-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hidrologi-api.service
sudo systemctl start hidrologi-api.service
```

### 3. **configs/logrotate.conf**
Log rotation configuration:
- Rotate logs daily
- Keep 30 days of logs
- Compress old logs
- Auto-cleanup

**Install:**
```bash
sudo cp configs/logrotate.conf /etc/logrotate.d/hidrologi-api
```

### 4. **.env.production.example**
Template untuk production environment variables:
- API configuration
- Paths
- Security tokens (harus diganti!)
- Performance settings
- Rate limiting

**Setup:**
```bash
cp .env.production.example .env.production
nano .env.production  # Edit dengan nilai sebenarnya
```

---

## 📋 Checklist Deployment

### Persiapan (Di Local)
- [ ] Pastikan semua file deployment ada di repository
- [ ] Buat Git repository (GitHub/GitLab/Bitbucket)
- [ ] Push semua code ke repository

### Setup VPS
- [ ] Pesan VPS (min 4GB RAM, Ubuntu 22.04)
- [ ] Setup DNS: A record `api.yourdomain.com` -> VPS IP
- [ ] Login ke VPS: `ssh root@YOUR_VPS_IP`

### Deployment
- [ ] Clone repository ke VPS atau upload via SCP
- [ ] Beri permission: `chmod +x scripts/*.sh`
- [ ] Jalankan: `sudo bash scripts/setup_vps.sh`
- [ ] Ikuti petunjuk (input domain & repository URL)
- [ ] Setup Earth Engine: `earthengine authenticate`
- [ ] Test API: `curl https://api.yourdomain.com/`

### Verifikasi
- [ ] Service running: `sudo systemctl status hidrologi-api`
- [ ] Nginx running: `sudo systemctl status nginx`
- [ ] SSL certificate installed: `sudo certbot certificates`
- [ ] API accessible dari external
- [ ] Test generate job berhasil
- [ ] Monitoring dashboard berjalan: `bash scripts/monitor.sh`

---

## 🚀 Quick Deploy Commands

```bash
# 1. Di VPS (sebagai root)
cd /root
git clone YOUR_REPO_URL project_hidrologi_ml
cd project_hidrologi_ml
chmod +x scripts/*.sh
sudo bash scripts/setup_vps.sh

# 2. Setelah setup selesai, switch ke user app
sudo su - hidrologi
cd project_hidrologi_ml
source venv/bin/activate
earthengine authenticate

# 3. Test API
curl https://api.yourdomain.com/
```

---

## 📊 Monitoring & Maintenance

### Daily Monitoring
```bash
# Real-time dashboard
bash scripts/monitor.sh

# Check logs
sudo journalctl -u hidrologi-api -f
tail -f ~/project_hidrologi_ml/logs/api.log
```

### Update Application
```bash
# Setelah push changes ke Git
bash scripts/update_app.sh
```

### Backup
```bash
# Manual backup
bash scripts/backup.sh

# Auto backup (setup cron)
crontab -e
# Add: 0 3 * * * /home/hidrologi/project_hidrologi_ml/scripts/backup.sh
```

---

## 🔐 Security Notes

1. **Ganti API Token!**
   - File: `.env.production`
   - Generate dengan: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

2. **Ganti Secret Key!**
   - File: `.env.production`
   - Generate dengan: `python3 -c "import secrets; print(secrets.token_hex(32))"`

3. **Protect .env file**
   ```bash
   chmod 600 .env.production
   ```

4. **Firewall aktif**
   ```bash
   sudo ufw status
   ```

---

## 📞 Troubleshooting

### Service tidak start
```bash
sudo journalctl -u hidrologi-api -n 50
cat ~/project_hidrologi_ml/logs/api_error.log
```

### Nginx error
```bash
sudo nginx -t
sudo tail -f /var/log/nginx/hidrologi_error.log
```

### Earth Engine error
```bash
sudo su - hidrologi
cd project_hidrologi_ml
source venv/bin/activate
earthengine authenticate
```

### Port sudah digunakan
```bash
sudo lsof -i :8001
sudo kill -9 PID
```

---

## 📚 File Structure Deployment

```
project_hidrologi_ml/
├── DEPLOYMENT_GUIDE.md          # Dokumentasi lengkap (15+ halaman)
├── QUICK_START.md               # Quick start guide
├── DEPLOYMENT_FILES.md          # This file - daftar semua files
│
├── scripts/                     # Automation scripts
│   ├── setup_vps.sh            # ⭐ Auto-setup VPS (main script)
│   ├── update_app.sh           # Update aplikasi
│   ├── monitor.sh              # Real-time monitoring
│   └── backup.sh               # Automated backup
│
├── configs/                     # Configuration files
│   ├── nginx.conf              # Nginx reverse proxy config
│   ├── hidrologi-api.service   # Systemd service config
│   ├── logrotate.conf          # Log rotation config
│   └── (files lain...)
│
├── .env.production.example      # Template environment production
│
└── project_hidrologi_ml/        # Application code
    ├── api_server.py
    ├── config.py
    └── ...
```

---

## ✅ Yang Sudah Otomatis Di-Handle

Script `setup_vps.sh` sudah menghandle:
- ✅ Update sistem
- ✅ Install semua dependencies (Python, GDAL, Nginx, dll)
- ✅ Buat user aplikasi dengan permissions
- ✅ Clone repository
- ✅ Setup virtual environment
- ✅ Install Python packages
- ✅ Generate secure tokens
- ✅ Buat direktori (results, temp, logs)
- ✅ Configure Nginx
- ✅ Setup Systemd service
- ✅ Configure firewall (UFW)
- ✅ Setup log rotation
- ✅ Install SSL certificate (Let's Encrypt)
- ✅ Auto-start service on boot

**Yang perlu manual:**
- Setup Google Earth Engine authentication (security requirement)
- Update DNS A record domain ke VPS IP

---

## 🎯 Next Steps After Deployment

1. **Monitoring**: Setup Grafana/Prometheus untuk monitoring advanced
2. **Backup**: Setup remote backup ke S3/Cloud Storage
3. **CI/CD**: Setup GitHub Actions untuk auto-deployment
4. **Load Balancer**: Jika traffic tinggi, setup multiple instances
5. **Database**: Jika perlu, setup PostgreSQL untuk job tracking

---

**Dokumentasi ini mencakup semua yang dibutuhkan untuk deployment production-ready! 🚀**

Lihat `QUICK_START.md` untuk langkah cepat, atau `DEPLOYMENT_GUIDE.md` untuk dokumentasi lengkap.
