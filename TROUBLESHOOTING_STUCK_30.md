# 🔧 Troubleshooting Guide: Stuck at 30%

## 📍 Problem: Processing stuck at 30% and not progressing

### 🎯 Root Cause
The 30% mark is where the API starts **fetching satellite data from Google Earth Engine (GEE)**. This is the most time-consuming and network-intensive operation.

---

## 🔍 Diagnosis Steps

### Step 1: Check if it's actually stuck or just slow
```bash
# Check live logs to see if there's activity
sudo supervisorctl tail -f hidrologi-api

# Or check the log file
sudo tail -f /var/log/supervisor/hidrologi-api.out.log
```

**What to look for:**
- `Loading main_weap_ml module...` - Module loading
- `CRITICAL SECTION: Google Earth Engine Data Fetching` - GEE fetch starting
- Any error messages about authentication or timeouts

**Expected behavior:**
- GEE fetching can take **2-10 minutes** depending on date range
- Longer date ranges = longer processing time
- No progress updates during GEE fetch (by design)

---

### Step 2: Check GEE Authentication
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana

# Test GEE auth
sudo -u www-data bash -c "source venv/bin/activate && python3 << 'EOF'
import ee
import json

creds_file = 'gee-credentials.json'
with open(creds_file) as f:
    creds = json.load(f)

credentials = ee.ServiceAccountCredentials(
    creds['client_email'],
    creds_file
)
ee.Initialize(credentials, project=creds['project_id'])

# Test query
image = ee.Image('UCSB-CHG/CHIRPS/DAILY/20230101')
info = image.getInfo()
print('SUCCESS: GEE Authentication working!')
print(f'Project: {creds[\"project_id\"]}')
EOF
"
```

**If this fails:**
- ✗ GEE credentials are invalid
- ✗ Service account doesn't have Earth Engine API access
- ✗ Project ID is incorrect

---

### Step 3: Check System Resources
```bash
# Memory
free -h

# CPU
top -bn1 | head -20

# Disk space
df -h

# Network
ping -c 3 earthengine.googleapis.com
```

**Common issues:**
- ❌ Out of memory (< 500MB free)
- ❌ High CPU usage (> 95%)
- ❌ No network connectivity to GEE servers

---

## 🛠️ Quick Fixes

### Fix 1: Run Auto-Fix Script
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo bash fix-issues.sh
```

This will:
- ✅ Fix permissions
- ✅ Update Python packages
- ✅ Test GEE authentication
- ✅ Restart service
- ✅ Clean old results

---

### Fix 2: Manual GEE Credentials Check
```bash
# Check if file exists
ls -lh /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json

# Check permissions
stat /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json

# Should show:
# - Owner: www-data
# - Permissions: 600 (rw-------)

# Fix if needed
sudo chown www-data:www-data /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json
sudo chmod 600 /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json
```

---

### Fix 3: Update .env.production
```bash
sudo nano /var/www/itriverdna.my.id/public_html/py_rivana/.env.production
```

Verify these settings:
```env
GEE_PROJECT_ID=your-actual-project-id
GEE_SERVICE_ACCOUNT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_FILE=/var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json
```

**Must match your actual GEE project!**

---

### Fix 4: Reinstall Earth Engine API
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade earthengine-api google-auth google-auth-oauthlib google-auth-httplib2"

# Restart service
sudo supervisorctl restart hidrologi-api:*
```

---

### Fix 5: Increase Memory (if low)
```bash
# Add 2GB swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

---

### Fix 6: Restart Everything
```bash
# Stop API
sudo supervisorctl stop hidrologi-api:*

# Wait 5 seconds
sleep 5

# Start API
sudo supervisorctl start hidrologi-api:*

# Check status
sudo supervisorctl status

# Test
curl http://localhost:8080/health
```

---

## 🧪 Test with Smaller Date Range

If stuck with large date range (e.g., 1 year), try smaller range first:

```bash
# From Laravel or curl
curl -X POST https://itriverdna.my.id/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": 110.4203,
    "latitude": -7.7956,
    "start": "2023-01-01",
    "end": "2023-01-31"
  }'
```

**Test progression:**
1. ✅ 1 month (30 days) - Should complete in 3-5 minutes
2. ✅ 3 months (90 days) - Should complete in 5-8 minutes
3. ✅ 6 months (180 days) - Should complete in 8-12 minutes
4. ✅ 1 year (365 days) - Should complete in 15-25 minutes

---

## 📊 Monitor Progress

### Real-time monitoring
```bash
# Terminal 1: Watch logs
sudo supervisorctl tail -f hidrologi-api

# Terminal 2: Watch system resources
watch -n 2 'free -h; echo ""; ps aux | grep "[a]pi_server.py" | head -3'

# Terminal 3: Monitor API status
watch -n 5 'curl -s http://localhost:8080/status/YOUR_JOB_ID | python3 -m json.tool'
```

---

## 🔍 Debug Script

Run comprehensive debug:
```bash
cd /var/www/itriverdna.my.id/public_html/py_rivana
sudo bash debug.sh
```

This will check:
- ✅ Service status
- ✅ Error logs
- ✅ GEE credentials
- ✅ Environment config
- ✅ System resources
- ✅ GEE connectivity

---

## 📝 Common Error Messages

### Error: "Could not authenticate"
**Cause:** Invalid GEE credentials

**Fix:**
```bash
# Re-upload credentials
scp gee-credentials.json root@vps:/var/www/itriverdna.my.id/public_html/py_rivana/

# Fix permissions
sudo chown www-data:www-data /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json
sudo chmod 600 /var/www/itriverdna.my.id/public_html/py_rivana/gee-credentials.json

# Restart
sudo supervisorctl restart hidrologi-api:*
```

---

### Error: "Quota exceeded"
**Cause:** GEE daily quota reached

**Fix:**
- Wait 24 hours for quota reset
- Use different GEE project
- Reduce date range

---

### Error: "Connection timeout"
**Cause:** Network issues or GEE servers slow

**Fix:**
```bash
# Test connectivity
ping -c 5 earthengine.googleapis.com

# Check if firewall blocking
sudo ufw status

# Allow outbound HTTPS
sudo ufw allow out 443/tcp

# Restart and retry
sudo supervisorctl restart hidrologi-api:*
```

---

### Error: "Memory error"
**Cause:** Insufficient RAM

**Fix:**
```bash
# Add swap (see Fix 5 above)
# Or reduce date range
# Or upgrade VPS RAM
```

---

## ✅ Verification Checklist

After fixes, verify:

- [ ] GEE credentials file exists and is readable
- [ ] .env.production has correct GEE settings
- [ ] GEE authentication test passes
- [ ] API service is running
- [ ] API responds to `/health` endpoint
- [ ] Memory > 500MB free
- [ ] Disk space > 10% free
- [ ] Network can reach earthengine.googleapis.com
- [ ] Test job with small date range completes

---

## 📞 Still Stuck?

### Check specific job logs
```bash
# Find your job ID
ls -lt /var/www/itriverdna.my.id/public_html/py_rivana/results/

# View job log
sudo cat /var/www/itriverdna.my.id/public_html/py_rivana/results/YOUR_JOB_ID/process.log

# View error log (if exists)
sudo cat /var/www/itriverdna.my.id/public_html/py_rivana/results/YOUR_JOB_ID/error.log
```

### Enable detailed logging
Edit `api_server.py` to add more debug output (already added in latest version).

### Contact Support
Provide:
1. Output of `sudo bash debug.sh`
2. Last 50 lines of error log
3. Job parameters (date range, coordinates)
4. VPS specs (RAM, CPU)

---

## 🎯 Prevention

To avoid future issues:

1. **Test with small date ranges first**
   - Start with 1 month
   - Gradually increase if successful

2. **Monitor resources**
   ```bash
   # Run monitoring script
   bash monitor.sh
   ```

3. **Regular maintenance**
   ```bash
   # Weekly cleanup
   find /var/www/itriverdna.my.id/public_html/py_rivana/results/* -mtime +7 -type d -exec rm -rf {} +
   
   # Monthly package updates
   cd /var/www/itriverdna.my.id/public_html/py_rivana
   sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade -r requirements.txt"
   ```

4. **Keep credentials valid**
   - GEE service account keys don't expire
   - But project access can be revoked
   - Verify periodically with test script

---

## 📊 Expected Timeline

| Date Range | Expected Processing Time | GEE Fetch Time |
|------------|-------------------------|----------------|
| 30 days    | 3-5 minutes             | 1-2 minutes    |
| 90 days    | 5-8 minutes             | 2-4 minutes    |
| 180 days   | 8-12 minutes            | 4-6 minutes    |
| 365 days   | 15-25 minutes           | 8-12 minutes   |

**Note:** Times vary based on:
- Server location
- Internet speed
- GEE server load
- VPS resources

---

**If stuck at 30% for more than 10 minutes with no log activity, it's definitely stuck. Run the debug script and apply fixes above.**
