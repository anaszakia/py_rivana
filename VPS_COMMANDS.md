# 🚀 Quick Commands for VPS

## 📍 Project Location
```
/var/www/itriverdna.my.id/public_html/py_rivana
```

---

## 🔍 **STUCK AT 30%? Run This First:**

```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo bash fix-issues.sh
```

This will automatically:
- ✅ Fix permissions
- ✅ Update packages
- ✅ Test GEE auth
- ✅ Restart service
- ✅ Test API

---

## 📊 Monitor Real-Time

```bash
# Watch logs live
sudo supervisorctl tail -f hidrologi-api

# Or
sudo tail -f /var/log/supervisor/hidrologi-api.out.log
```

---

## 🔧 Common Commands

### Service Management
```bash
# Status
sudo supervisorctl status

# Restart
sudo supervisorctl restart hidrologi-api:*

# Stop
sudo supervisorctl stop hidrologi-api:*

# Start
sudo supervisorctl start hidrologi-api:*
```

### View Logs
```bash
# Live logs
sudo supervisorctl tail -f hidrologi-api

# Error logs
sudo tail -f /var/log/supervisor/hidrologi-api.err.log

# Last 100 lines
sudo tail -100 /var/log/supervisor/hidrologi-api.out.log
```

### Update Code
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo git pull origin main
sudo supervisorctl restart hidrologi-api:*
```

---

## 🧪 Test GEE Authentication

```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo -u www-data bash -c "source venv/bin/activate && python3 << 'EOF'
import ee
import json

with open('gee-credentials.json') as f:
    creds = json.load(f)

credentials = ee.ServiceAccountCredentials(
    creds['client_email'],
    'gee-credentials.json'
)
ee.Initialize(credentials, project=creds['project_id'])

image = ee.Image('UCSB-CHG/CHIRPS/DAILY/20230101')
info = image.getInfo()
print('✓ GEE Authentication SUCCESS!')
EOF
"
```

---

## 🔍 Debug Comprehensive

```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo bash debug.sh
```

---

## 🧹 Clean Old Results

```bash
# Clean results older than 7 days
find /var/www/itriverdna.my.id/public_html/py_rivana/results/* -mtime +7 -type d -exec rm -rf {} +
```

---

## 📈 System Resources

```bash
# Memory
free -h

# Disk
df -h

# CPU
top

# Processes
ps aux | grep api_server
```

---

## 🔐 Fix Permissions

```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo chown -R www-data:www-data .
sudo chmod -R 755 .
sudo chmod -R 775 results
sudo chmod 600 gee-credentials.json
```

---

## 📝 View Job Logs

```bash
# List recent jobs
ls -lt /var/www/itriverdna.my.id/public_html/py_rivana/results/ | head -10

# View specific job log
sudo cat /var/www/itriverdna.my.id/public_html/py_rivana/results/JOB_ID/process.log

# View error log (if exists)
sudo cat /var/www/itriverdna.my.id/public_html/py_rivana/results/JOB_ID/error.log
```

---

## 🧪 Test API

```bash
# Health check
curl http://localhost:8080/health

# Test job (small date range)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": 110.4203,
    "latitude": -7.7956,
    "start": "2023-01-01",
    "end": "2023-01-31"
  }'
```

---

## ⚡ Quick Fix Commands

```bash
# 1. If API not responding
sudo supervisorctl restart hidrologi-api:*
sleep 3
curl http://localhost:8080/health

# 2. If out of memory
free -h
# Add swap if needed (see TROUBLESHOOTING_STUCK_30.md)

# 3. If disk full
df -h
find /var/www/itriverdna.my.id/public_html/py_rivana/results/* -mtime +3 -type d -exec rm -rf {} +

# 4. If GEE auth fails
cd /var/www/itriverdna.my.id/public_html/py_rivana
ls -lh gee-credentials.json
sudo chmod 600 gee-credentials.json
sudo chown www-data:www-data gee-credentials.json

# 5. If stuck at 30%
sudo bash fix-issues.sh
```

---

## 📊 Check Job Status

```bash
# Via API
curl http://localhost:8080/status/YOUR_JOB_ID | python3 -m json.tool

# Or check files directly
ls -lh /var/www/itriverdna.my.id/public_html/py_rivana/results/YOUR_JOB_ID/
```

---

## 🔄 Complete Restart

```bash
# Stop everything
sudo supervisorctl stop hidrologi-api:*

# Wait
sleep 5

# Start
sudo supervisorctl start hidrologi-api:*

# Check
sudo supervisorctl status
curl http://localhost:8080/health
```

---

## 📚 Documentation

- **Full Deployment**: `DEPLOYMENT_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **Stuck at 30%**: `TROUBLESHOOTING_STUCK_30.md`

---

## 🆘 Emergency

If nothing works:

```bash
# Full system check
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo bash debug.sh > debug-output.txt 2>&1

# Send debug-output.txt for support
cat debug-output.txt
```

---

**Most common fix for "stuck at 30%":**
```bash
sudo bash fix-issues.sh
```
