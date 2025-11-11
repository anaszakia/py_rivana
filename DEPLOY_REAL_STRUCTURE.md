# 🚀 PANDUAN DEPLOY UNTUK STRUKTUR DIRECTORY ANDA

Berdasarkan screenshot clone Anda di VPS, struktur directory adalah:
```
/var/www/itriverdna.my.id/public_html/py_rivana/
├── project_hidrologi_ml/
│   ├── api_server.py
│   ├── config.py
│   └── ...
├── scripts/
├── configs/
├── requirements.txt
└── ...
```

## 📋 Langkah Deploy (Disesuaikan)

### 1. **Sudah Clone ✅** 
Anda sudah di tahap ini:
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
```

### 2. **Beri Permission ke Scripts**
```bash
chmod +x scripts/*.sh
```

### 3. **Jalankan Setup Script V2 (yang sudah disesuaikan)**
```bash
sudo bash scripts/setup_vps_v2.sh
```

Script akan:
- ✅ Auto-detect directory structure Anda
- ✅ Menemukan `project_hidrologi_ml/api_server.py`
- ✅ Setup semuanya berdasarkan path actual

Saat ditanya:
- **Domain**: Masukkan `api.itriverdna.my.id` atau subdomain yang Anda inginkan
- Script akan generate tokens otomatis

### 4. **Setup Earth Engine**
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
source venv/bin/activate
earthengine authenticate
```

### 5. **Test API**
```bash
# Test local
curl http://localhost:8001/

# Atau jika sudah SSL
curl https://api.itriverdna.my.id/
```

---

## 🔧 Alternative: Setup Manual (Jika Script Bermasalah)

Jika ada masalah dengan script, ikuti ini:

### A. Setup Virtual Environment
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn python-dotenv
```

### B. Buat .env.production
```bash
nano .env.production
```

Isi dengan:
```ini
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8001
DEBUG=False
RESULTS_DIR=/var/www/itriverdna.my.id/public_html/py_rivana/results
TEMP_DIR=/var/www/itriverdna.my.id/public_html/py_rivana/temp
EE_AUTHENTICATED=True
SECRET_KEY=GANTI_DENGAN_TOKEN_RANDOM
API_TOKEN=GANTI_DENGAN_TOKEN_RANDOM
LOG_FILE=/var/www/itriverdna.my.id/public_html/py_rivana/logs/api.log
```

Generate tokens:
```bash
python3 -c "import secrets; print('SECRET_KEY:', secrets.token_hex(32))"
python3 -c "import secrets; print('API_TOKEN:', secrets.token_urlsafe(32))"
```

### C. Buat Directories
```bash
mkdir -p results temp logs
chmod 755 results temp logs
```

### D. Test Run Manual
```bash
source venv/bin/activate
cd /var/www/itriverdna.my.id/public_html/py_rivana
python3 project_hidrologi_ml/api_server.py
```

Jika jalan (Ctrl+C untuk stop), lanjut setup systemd.

### E. Setup Systemd Service
```bash
sudo nano /etc/systemd/system/hidrologi-api.service
```

Isi:
```ini
[Unit]
Description=Hidrologi ML API Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/itriverdna.my.id/public_html/py_rivana
Environment="PATH=/var/www/itriverdna.my.id/public_html/py_rivana/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=/var/www/itriverdna.my.id/public_html/py_rivana/venv/bin/python3 /var/www/itriverdna.my.id/public_html/py_rivana/project_hidrologi_ml/api_server.py
Restart=always
RestartSec=10
StandardOutput=append:/var/www/itriverdna.my.id/public_html/py_rivana/logs/api.log
StandardError=append:/var/www/itriverdna.my.id/public_html/py_rivana/logs/api_error.log

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hidrologi-api.service
sudo systemctl start hidrologi-api.service
sudo systemctl status hidrologi-api.service
```

### F. Setup Nginx
```bash
sudo nano /etc/nginx/sites-available/hidrologi-api
```

Isi:
```nginx
upstream hidrologi_api {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name api.itriverdna.my.id;
    
    location / {
        proxy_pass http://hidrologi_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/hidrologi-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### G. Setup SSL
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d api.itriverdna.my.id
```

---

## ✅ Verifikasi

### 1. Check Service
```bash
sudo systemctl status hidrologi-api.service
```

### 2. Check Logs
```bash
tail -f /var/www/itriverdna.my.id/public_html/py_rivana/logs/api.log
```

### 3. Test API
```bash
curl http://localhost:8001/
```

### 4. Test dengan Token
```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     http://localhost:8001/jobs
```

---

## 🐛 Troubleshooting

### Service tidak start?
```bash
# Check detail error
sudo journalctl -u hidrologi-api.service -n 50

# Check Python path
which python3
/var/www/itriverdna.my.id/public_html/py_rivana/venv/bin/python3 --version

# Check api_server.py
ls -la /var/www/itriverdna.my.id/public_html/py_rivana/project_hidrologi_ml/api_server.py
```

### Port 8001 sudah dipakai?
```bash
sudo lsof -i :8001
sudo kill -9 PID_YANG_MUNCUL
```

### Module not found?
```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## 📞 Next Steps

Setelah semua jalan:

1. **Setup Earth Engine** (WAJIB)
2. **Test generate job**
3. **Setup monitoring**
4. **Setup backup**

Gunakan script yang sudah disesuaikan di `scripts/setup_vps_v2.sh` untuk proses otomatis! 🚀
