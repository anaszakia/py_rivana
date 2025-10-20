# 🚀 DEPLOYMENT GUIDE - Environment Configuration

## 📋 Overview

Project ini mendukung 2 environment:
- **Local Development** (Windows/Mac/Linux)
- **Production** (VPS Ubuntu/Debian)

Configuration otomatis detect environment berdasarkan OS dan paths.

---

## 🔧 LOCAL DEVELOPMENT

### Setup:

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file (sudah ada .env default)
# File .env sudah di-set untuk local development

# 4. Run server
cd project_hidrologi_ml
python api_server.py

# Server akan jalan di: http://127.0.0.1:8000
```

### Configuration File: `.env.local`

```env
API_HOST=127.0.0.1      # Localhost only
API_PORT=8000           # Default port
DEBUG=True              # Debug mode aktif
RESULTS_DIR=results     # Relative path (akan auto-resolve)
```

### Features:
- ✅ Debug mode enabled
- ✅ Detailed logging
- ✅ CORS enabled
- ✅ Auto-reload pada code changes (manual restart)
- ✅ Results tersimpan di `results/` folder local

---

## 🌐 PRODUCTION (VPS)

### Setup di VPS:

```bash
# 1. Clone/upload project
cd /home/hidrologi
git clone <your-repo>
cd project_hidrologi_ml

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy production environment file
cp project_hidrologi_ml/.env.production project_hidrologi_ml/.env

# 5. EDIT .env file - GANTI SECRET_KEY!
nano project_hidrologi_ml/.env

# Generate secure secret key:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Paste ke .env:
SECRET_KEY=<generated_key_here>

# 6. Test run
cd project_hidrologi_ml
python api_server.py

# Server akan jalan di: http://0.0.0.0:8000
```

### Configuration File: `.env.production`

```env
API_HOST=0.0.0.0               # Listen on all interfaces
API_PORT=8000                  # Production port
DEBUG=False                    # Debug disabled
RESULTS_DIR=/home/hidrologi/project_hidrologi_ml/results  # Absolute path
LOG_FILE=/var/log/hidrologi/api.log
```

### Features:
- ✅ Production-ready
- ✅ Security hardened
- ✅ Efficient logging
- ✅ Rate limiting support
- ✅ Absolute paths

---

## 🔄 Switching Environments

### Method 1: Auto-Detection (Recommended)

Config otomatis detect environment:
- Windows → Local
- Linux dengan path `/home/hidrologi` → Production

Tidak perlu konfigurasi tambahan!

### Method 2: Environment Variable

```bash
# Local
export ENVIRONMENT=local
python api_server.py

# Production
export ENVIRONMENT=production
python api_server.py
```

### Method 3: Symlink .env

```bash
# Use local
ln -sf .env.local .env

# Use production
ln -sf .env.production .env
```

---

## 📊 Configuration Manager

File `config.py` handle semua configuration:

```python
from config import config

# Access configuration
print(config.API_PORT)          # 8000
print(config.RESULTS_DIR)       # Auto-resolved path
print(config.is_production())   # True/False
print(config.DEBUG)             # True/False

# Print full config
config.print_config()
```

### Auto-Features:

✅ **Auto-detect environment** (Windows = local, Linux = production)
✅ **Auto-create directories** (results, temp, logs)
✅ **Path resolution** (relative paths → absolute paths)
✅ **Validation** (check required settings)

---

## 🛠️ Running in Different Modes

### Local Development:

```powershell
# Default (port 8000, localhost)
python api_server.py

# Custom port
python api_server.py 8080

# Custom host and port
python api_server.py 8080 127.0.0.1
```

### Production (Manual):

```bash
# Default
python api_server.py

# Custom port
python api_server.py 8000

# All interfaces
python api_server.py 8000 0.0.0.0
```

### Production (Supervisor):

```ini
# /etc/supervisor/conf.d/hidrologi-api.conf
[program:hidrologi-api]
command=/home/hidrologi/project_hidrologi_ml/venv/bin/python \
        /home/hidrologi/project_hidrologi_ml/project_hidrologi_ml/api_server.py
directory=/home/hidrologi/project_hidrologi_ml/project_hidrologi_ml
user=hidrologi
autostart=true
autorestart=true
```

---

## 🔍 Troubleshooting

### "Config not loaded" warning:

```bash
# Install python-dotenv
pip install python-dotenv
```

### Wrong environment detected:

```bash
# Force environment
export ENVIRONMENT=production
python api_server.py
```

### Directory creation failed:

```bash
# Check permissions
ls -la /home/hidrologi/project_hidrologi_ml/

# Fix ownership
sudo chown -R hidrologi:hidrologi /home/hidrologi/project_hidrologi_ml/
```

### Port already in use:

```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8000

# Linux:
lsof -i :8000

# Kill process or use different port
python api_server.py 8001
```

---

## 📝 Environment Variables Reference

| Variable | Local | Production | Description |
|----------|-------|------------|-------------|
| `API_HOST` | 127.0.0.1 | 0.0.0.0 | Bind address |
| `API_PORT` | 8000 | 8000 | Listen port |
| `DEBUG` | True | False | Debug mode |
| `ENVIRONMENT` | local | production | Environment name |
| `RESULTS_DIR` | results | /home/.../results | Results storage |
| `TEMP_DIR` | temp | /tmp/hidrologi | Temp storage |
| `LOG_FILE` | logs/api.log | /var/log/hidrologi/api.log | Log file path |
| `LOG_LEVEL` | DEBUG | INFO | Logging level |
| `MAX_CONCURRENT_JOBS` | 2 | 5 | Job limit |
| `SECRET_KEY` | dev_key | <secure_key> | Security key |

---

## ✅ Checklist

### Local Setup:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file exists (or using default `.env.local`)
- [ ] Server starts without errors
- [ ] Can access http://localhost:8000

### Production Setup:
- [ ] VPS access configured
- [ ] Project uploaded/cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `.env.production` copied to `.env`
- [ ] **SECRET_KEY changed to secure random string**
- [ ] Log directories created with proper permissions
- [ ] Server runs manually without errors
- [ ] Supervisor configured (for auto-restart)
- [ ] Nginx configured (for reverse proxy)
- [ ] SSL certificate installed (Certbot)
- [ ] Firewall rules configured (UFW)

---

## 🎉 Success!

Jika server berhasil start, Anda akan lihat output:

```
================================
🔧 CONFIGURATION - LOCAL MODE
================================
Environment:     local
Debug Mode:      True
API Host:        127.0.0.1:8000
Results Dir:     E:\laragon\www\project_hidrologi_ml\results
...
================================

✅ Server berhasil started di http://127.0.0.1:8000
🌍 Environment: LOCAL
📊 Debug Mode: True
📁 Results Dir: E:\laragon\www\project_hidrologi_ml\results
📡 Listening for requests...
```

Test endpoint:
```bash
curl http://localhost:8000/jobs
```

Expected response:
```json
{"jobs": []}
```
