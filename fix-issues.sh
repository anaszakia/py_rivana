#!/bin/bash

# ============================================
# Fix Common Issues Script
# For "Stuck at 30%" problem
# ============================================

echo "🔧 Fixing Common Issues - Hidrologi ML API"
echo "==========================================="
echo ""

PROJECT_DIR="/var/www/itriverdna.my.id/public_html/py_rivana"

if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root: sudo bash fix-issues.sh"
    exit 1
fi

# 1. Check and fix permissions
echo "1. Fixing file permissions..."
chown -R www-data:www-data "$PROJECT_DIR"
chmod -R 755 "$PROJECT_DIR"
chmod -R 775 "$PROJECT_DIR/results"
if [ -f "$PROJECT_DIR/gee-credentials.json" ]; then
    chmod 600 "$PROJECT_DIR/gee-credentials.json"
    chown www-data:www-data "$PROJECT_DIR/gee-credentials.json"
    echo "   ✓ GEE credentials permissions fixed"
fi
echo "   ✓ Permissions fixed"
echo ""

# 2. Check Python packages
echo "2. Checking Python packages..."
cd "$PROJECT_DIR"
if [ -d "venv" ]; then
    echo "   Installing/updating packages..."
    sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade earthengine-api google-auth google-auth-oauthlib google-auth-httplib2"
    echo "   ✓ Packages updated"
else
    echo "   ✗ Virtual environment not found!"
fi
echo ""

# 3. Test GEE authentication
echo "3. Testing GEE authentication..."
sudo -u www-data bash -c "cd '$PROJECT_DIR' && source venv/bin/activate && python3 << 'EOF'
import ee
import json
import os

try:
    creds_file = '$PROJECT_DIR/gee-credentials.json'
    if os.path.exists(creds_file):
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
        print('   ✓ GEE Authentication SUCCESS')
        print(f'   Project: {creds[\"project_id\"]}')
    else:
        print('   ✗ GEE credentials file not found!')
except Exception as e:
    print(f'   ✗ GEE Authentication FAILED: {str(e)}')
EOF
"
echo ""

# 4. Check memory
echo "4. Checking system resources..."
TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
USED_MEM=$(free -m | awk 'NR==2{print $3}')
FREE_MEM=$(free -m | awk 'NR==2{print $4}')

echo "   Memory: ${USED_MEM}MB used / ${TOTAL_MEM}MB total (${FREE_MEM}MB free)"

if [ "$FREE_MEM" -lt 500 ]; then
    echo "   ⚠️  WARNING: Low memory (< 500MB free)"
    echo "   Consider adding swap or restarting services"
fi
echo ""

# 5. Check disk space
echo "5. Checking disk space..."
DISK_USAGE=$(df -h "$PROJECT_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
DISK_AVAIL=$(df -h "$PROJECT_DIR" | awk 'NR==2{print $4}')

echo "   Disk usage: ${DISK_USAGE}% (${DISK_AVAIL} available)"

if [ "$DISK_USAGE" -gt 90 ]; then
    echo "   ⚠️  WARNING: Low disk space"
    echo "   Cleaning old results..."
    find "$PROJECT_DIR/results/"* -mtime +7 -type d -exec rm -rf {} + 2>/dev/null
    echo "   ✓ Old results cleaned"
fi
echo ""

# 6. Restart service
echo "6. Restarting API service..."
supervisorctl restart hidrologi-api:*
sleep 3
supervisorctl status hidrologi-api:*
echo ""

# 7. Test API
echo "7. Testing API endpoint..."
sleep 2
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8080/health)
if [ "$HTTP_CODE" == "200" ]; then
    echo "   ✓ API is responding (HTTP $HTTP_CODE)"
else
    echo "   ✗ API is not responding (HTTP $HTTP_CODE)"
    echo "   Check logs: sudo supervisorctl tail -f hidrologi-api"
fi
echo ""

# 8. Show recent logs
echo "8. Recent error logs (last 20 lines):"
echo "======================================"
tail -20 /var/log/supervisor/hidrologi-api.err.log 2>/dev/null || echo "No error logs"
echo ""

echo "==========================================="
echo "Fix complete! Common issues addressed:"
echo "  ✓ Permissions fixed"
echo "  ✓ Python packages updated"
echo "  ✓ GEE authentication tested"
echo "  ✓ Resources checked"
echo "  ✓ Service restarted"
echo ""
echo "If still stuck at 30%, check:"
echo "  1. GEE credentials are valid"
echo "  2. GEE project has Earth Engine API enabled"
echo "  3. Network can reach earthengine.googleapis.com"
echo "  4. Date range is not too large (try 1 month first)"
echo ""
echo "View live logs:"
echo "  sudo supervisorctl tail -f hidrologi-api"
echo ""
