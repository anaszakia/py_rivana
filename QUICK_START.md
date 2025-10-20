# 🚀 Quick Deployment Guide

## 📦 One-Line Deployment

Setelah clone repository ke VPS, jalankan:

```bash
cd /var/www/itriverdna.my.id/public_html/project_hidrologi_ml
sudo bash setup.sh
```

## 🔑 Pre-requisites

1. **VPS Requirements:**
   - Ubuntu 20.04/22.04
   - Minimal 4GB RAM
   - 20GB Disk Space
   - Root/sudo access

2. **Domain Setup:**
   - Point `itriverdna.my.id` ke IP VPS
   - Point `api.itriverdna.my.id` ke IP VPS (optional)

3. **Google Earth Engine:**
   - Service Account JSON credentials
   - Project ID

## 📝 Step-by-Step

### 1. Clone Repository
```bash
cd /var/www/itriverdna.my.id/public_html
git clone https://github.com/YOUR_USERNAME/project_hidrologi_ml.git
cd project_hidrologi_ml
```

### 2. Run Setup Script
```bash
sudo bash setup.sh
```

### 3. Upload GEE Credentials
```bash
# From local machine:
scp gee-credentials.json root@your-vps:/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/

# On server:
sudo chmod 600 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
sudo chown www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/gee-credentials.json
```

### 4. Configure Environment
```bash
sudo nano /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/.env.production
```

Update:
- `GEE_PROJECT_ID`
- `GEE_SERVICE_ACCOUNT_EMAIL`

### 5. Start Service
```bash
sudo supervisorctl start hidrologi-api:*
```

### 6. Setup SSL (Optional but Recommended)
```bash
sudo certbot --nginx -d itriverdna.my.id -d api.itriverdna.my.id
```

### 7. Test API
```bash
curl https://itriverdna.my.id/api/health
```

## 🔄 Regular Operations

### Deploy Updates
```bash
sudo bash deploy.sh
```

### Monitor Status
```bash
bash monitor.sh
```

### View Logs
```bash
# Real-time logs
sudo supervisorctl tail -f hidrologi-api

# Error logs
sudo tail -f /var/log/supervisor/hidrologi-api.err.log

# Nginx logs
sudo tail -f /var/log/nginx/hidrologi-api-error.log
```

### Restart Service
```bash
sudo supervisorctl restart hidrologi-api:*
```

### Check Status
```bash
sudo supervisorctl status
```

## 🛠️ Troubleshooting

### API Not Starting

```bash
# Check logs
sudo supervisorctl tail hidrologi-api stderr

# Check Python version
python3.11 --version

# Reinstall dependencies
cd /var/www/itriverdna.my.id/public_html/project_hidrologi_ml
sudo -u www-data bash -c "source venv/bin/activate && pip install -r requirements.txt"
```

### Port Already in Use

```bash
# Find process using port 8080
sudo netstat -tulpn | grep 8080

# Kill process
sudo kill -9 PID
```

### Permission Denied

```bash
# Fix permissions
sudo chown -R www-data:www-data /var/www/itriverdna.my.id/public_html/project_hidrologi_ml
sudo chmod -R 755 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml
sudo chmod -R 775 /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results
```

### High Memory Usage

```bash
# Check memory
free -h

# Restart service
sudo supervisorctl restart hidrologi-api:*

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### System Status
```bash
# CPU
top -bn1 | grep "Cpu(s)"

# Memory
free -h

# Disk
df -h

# Processes
ps aux | grep api_server.py
```

### Logs Location
- Supervisor: `/var/log/supervisor/hidrologi-api*.log`
- Nginx: `/var/log/nginx/hidrologi-api*.log`
- Application: `/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/api_server.log`

## 🔐 Security

### Firewall
```bash
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Update SSL
```bash
# Manual renewal
sudo certbot renew

# Test renewal
sudo certbot renew --dry-run
```

## 📁 File Structure

```
/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/
├── project_hidrologi_ml/
│   ├── api_server.py          # Main API server
│   └── main_weap_ml.py        # ML processor
├── results/                    # Job results (auto-cleanup after 30 days)
├── venv/                       # Python virtual environment
├── .env.production            # Production config
├── gee-credentials.json       # GEE credentials (keep secure!)
├── requirements.txt           # Python dependencies
├── setup.sh                   # Initial setup script
├── deploy.sh                  # Deployment script
├── monitor.sh                 # Monitoring script
└── DEPLOYMENT_GUIDE.md       # Full documentation
```

## 🌐 API Endpoints

After deployment, API will be available at:

- **Base URL**: `https://itriverdna.my.id/api/`
- **Health Check**: `https://itriverdna.my.id/api/health`
- **Generate Job**: `POST https://itriverdna.my.id/api/generate`
- **Job Status**: `GET https://itriverdna.my.id/api/status/{job_id}`
- **Job Summary**: `GET https://itriverdna.my.id/api/summary/{job_id}`

## 🔄 Auto-Cleanup

Cron job automatically cleans results older than 30 days:
```bash
# View crontab
sudo crontab -l

# Manual cleanup
find /var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results/* -mtime +30 -type d -exec rm -rf {} +
```

## 📞 Support

For detailed documentation, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Common Commands
```bash
# Status
sudo supervisorctl status

# Start
sudo supervisorctl start hidrologi-api:*

# Stop
sudo supervisorctl stop hidrologi-api:*

# Restart
sudo supervisorctl restart hidrologi-api:*

# Logs
sudo supervisorctl tail -f hidrologi-api

# Monitor
bash monitor.sh

# Deploy update
sudo bash deploy.sh
```

## ✅ Checklist

- [ ] VPS prepared with Ubuntu
- [ ] Domain pointing to VPS IP
- [ ] Repository cloned to `/var/www/itriverdna.my.id/public_html/`
- [ ] Run `setup.sh`
- [ ] Upload GEE credentials
- [ ] Edit `.env.production`
- [ ] Start service with supervisorctl
- [ ] Setup SSL with certbot
- [ ] Test API endpoints
- [ ] Configure Laravel to use new API URL

## 🎯 Production Checklist

- [ ] API responding on HTTPS
- [ ] Supervisor auto-restart enabled
- [ ] Nginx reverse proxy configured
- [ ] SSL certificate valid
- [ ] Firewall rules set
- [ ] Auto-cleanup cron job active
- [ ] Monitoring script working
- [ ] Laravel .env updated with API URL

---

**Ready to deploy? Run: `sudo bash setup.sh` 🚀**
