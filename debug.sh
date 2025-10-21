#!/bin/bash

# ============================================
# Debug Script - Check API Issue at 30%
# ============================================

echo "🔍 Debugging Hidrologi ML API - Stuck at 30% Issue"
echo "=================================================="
echo ""

PROJECT_DIR="/var/www/itriverdna.my.id/public_html/py_rivana"

# 1. Check if service is running
echo "1. Checking Service Status..."
sudo supervisorctl status hidrologi-api:* 2>/dev/null || echo "Supervisor not configured"
echo ""

# 2. Check recent error logs
echo "2. Recent Error Logs (Last 50 lines):"
echo "======================================="
if [ -f /var/log/supervisor/hidrologi-api.err.log ]; then
    sudo tail -50 /var/log/supervisor/hidrologi-api.err.log
else
    echo "No error log found"
fi
echo ""

# 3. Check output logs
echo "3. Recent Output Logs (Last 30 lines):"
echo "======================================="
if [ -f /var/log/supervisor/hidrologi-api.out.log ]; then
    sudo tail -30 /var/log/supervisor/hidrologi-api.out.log
else
    echo "No output log found"
fi
echo ""

# 4. Check API process
echo "4. API Process Status:"
echo "======================"
ps aux | grep "[a]pi_server.py" || echo "No API process running"
echo ""

# 5. Check port status
echo "5. Port 8080 Status:"
echo "===================="
sudo netstat -tulpn | grep ":8080" || echo "Port 8080 not listening"
echo ""

# 6. Check GEE credentials
echo "6. GEE Credentials Check:"
echo "========================="
if [ -f "$PROJECT_DIR/gee-credentials.json" ]; then
    echo "✓ GEE credentials file exists"
    ls -lh "$PROJECT_DIR/gee-credentials.json"
else
    echo "✗ GEE credentials file NOT FOUND!"
fi
echo ""

# 7. Check environment file
echo "7. Environment Configuration:"
echo "============================="
if [ -f "$PROJECT_DIR/.env.production" ]; then
    echo "✓ .env.production exists"
    echo "GEE Config:"
    grep "GEE_" "$PROJECT_DIR/.env.production" | grep -v "PRIVATE_KEY"
else
    echo "✗ .env.production NOT FOUND!"
fi
echo ""

# 8. Check disk space
echo "8. Disk Space:"
echo "=============="
df -h "$PROJECT_DIR"
echo ""

# 9. Check memory
echo "9. Memory Usage:"
echo "================"
free -h
echo ""

# 10. Test GEE connection
echo "10. Testing GEE Connection:"
echo "==========================="
cd "$PROJECT_DIR"
sudo -u www-data bash -c "source venv/bin/activate && python3 -c '
import ee
import json
import os

try:
    # Try to read credentials
    creds_file = \"$PROJECT_DIR/gee-credentials.json\"
    if os.path.exists(creds_file):
        print(\"✓ Credentials file found\")
        with open(creds_file) as f:
            creds = json.load(f)
            print(f\"  Project: {creds.get(\"project_id\", \"N/A\")}\")
            print(f\"  Email: {creds.get(\"client_email\", \"N/A\")}\")
        
        # Try to authenticate
        credentials = ee.ServiceAccountCredentials(
            creds.get(\"client_email\"),
            creds_file
        )
        ee.Initialize(credentials, project=creds.get(\"project_id\"))
        print(\"✓ GEE Authentication successful\")
        
        # Try simple query
        image = ee.Image(\"UCSB-CHG/CHIRPS/DAILY/20230101\")
        info = image.getInfo()
        print(\"✓ GEE Data access successful\")
    else:
        print(\"✗ Credentials file not found\")
except Exception as e:
    print(f\"✗ GEE Error: {str(e)}\")
'" 2>&1
echo ""

# 11. Check recent jobs
echo "11. Recent Jobs:"
echo "================"
if [ -f "$PROJECT_DIR/jobs.json" ]; then
    echo "jobs.json exists, showing last 3 jobs:"
    sudo tail -20 "$PROJECT_DIR/jobs.json" | head -15
else
    echo "No jobs.json file found"
fi
echo ""

# 12. Check results directory
echo "12. Results Directory:"
echo "======================"
if [ -d "$PROJECT_DIR/results" ]; then
    echo "Total jobs: $(ls -1 $PROJECT_DIR/results | wc -l)"
    echo "Recent jobs:"
    ls -lht "$PROJECT_DIR/results" | head -5
else
    echo "Results directory not found"
fi
echo ""

echo "=================================================="
echo "Debug complete. Check output above for issues."
echo ""
echo "Common issues at 30% (GEE fetching):"
echo "1. GEE credentials not valid or expired"
echo "2. GEE project quota exceeded"
echo "3. Network timeout to GEE servers"
echo "4. Insufficient memory"
echo "5. Python packages not installed correctly"
echo ""
echo "To view live logs, run:"
echo "  sudo supervisorctl tail -f hidrologi-api"
echo ""
