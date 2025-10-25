#!/bin/bash

echo "================================================================================"
echo "🔧 FORCE UPDATE SERVER - Fix kapasitas_RETENSI Error"
echo "================================================================================"

# Navigate to project directory
cd /var/www/itriverdna.my.id/public_html/py_rivana

echo ""
echo "1️⃣  Stopping rivana-api service..."
sudo systemctl stop rivana-api
sleep 2

echo ""
echo "2️⃣  Killing any remaining Python processes..."
sudo pkill -9 -f "python.*rivana" || echo "   No processes to kill"
sudo pkill -9 -f "api_server" || echo "   No api_server processes"
sleep 1

echo ""
echo "3️⃣  Removing ALL Python cache files..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ Cache cleared"

echo ""
echo "4️⃣  Force pulling latest code from GitHub..."
git fetch origin
git reset --hard origin/main
echo "   ✅ Code updated to latest version"

echo ""
echo "5️⃣  Verifying fix in main_weap_ml.py..."
echo "   Checking line 79 (WEAPConfig):"
sed -n '79p' project_hidrologi_ml/main_weap_ml.py
echo ""
echo "   Checking line 1940 (usage):"
sed -n '1940p' project_hidrologi_ml/main_weap_ml.py
echo ""
echo "   Searching for any 'kapasitas_RETENSI' (should be NONE):"
grep -n "kapasitas_RETENSI" project_hidrologi_ml/main_weap_ml.py || echo "   ✅ No 'kapasitas_RETENSI' found (GOOD!)"

echo ""
echo "6️⃣  Removing old results..."
rm -rf results/*
echo "   ✅ Old results cleared"

echo ""
echo "7️⃣  Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "8️⃣  Starting rivana-api service..."
sudo systemctl start rivana-api
sleep 3

echo ""
echo "9️⃣  Checking service status..."
sudo systemctl status rivana-api --no-pager -l

echo ""
echo "🔟  Restarting nginx..."
sudo systemctl restart nginx

echo ""
echo "================================================================================"
echo "✅ UPDATE COMPLETE!"
echo "================================================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Wait 5 seconds for services to fully start"
echo "   2. Run a NEW analysis from Laravel web interface"
echo "   3. Check if error is resolved"
echo ""
echo "🔍 To verify service is running:"
echo "   sudo systemctl status rivana-api"
echo ""
echo "📊 To check logs:"
echo "   sudo journalctl -u rivana-api -f"
echo ""
echo "================================================================================"
