#!/bin/bash

# ============================================
# Monitoring Script - Hidrologi ML API
# ============================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}     ${BLUE}Hidrologi ML API Monitor${NC}         ${CYAN}║${NC}"
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo ""
}

print_section() {
    echo -e "${YELLOW}▶ $1${NC}"
    echo "─────────────────────────────────────────"
}

print_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

clear
print_header

# 1. Service Status
print_section "1. Service Status"
if command -v supervisorctl &> /dev/null; then
    supervisorctl status hidrologi-api:* 2>/dev/null
    if [ $? -eq 0 ]; then
        print_ok "Supervisor is managing the API"
    else
        print_error "Supervisor is not running or API not configured"
    fi
else
    print_error "Supervisor not installed"
fi
echo ""

# 2. API Health Check
print_section "2. API Health Check"
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8080/health 2>/dev/null)
if [ "$HTTP_CODE" == "200" ]; then
    print_ok "API is responding (HTTP $HTTP_CODE)"
    RESPONSE=$(curl -s http://localhost:8080/health 2>/dev/null)
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    print_error "API is not responding (HTTP ${HTTP_CODE:-'no response'})"
fi
echo ""

# 3. System Resources
print_section "3. System Resources"
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  " 100 - $1"%"}'

echo "Memory Usage:"
free -h | awk 'NR==2{printf "  Used: %s / %s (%.2f%%)\n", $3, $2, $3*100/$2}'

echo "Disk Usage:"
df -h / | awk 'NR==2{printf "  Used: %s / %s (%s)\n", $3, $2, $5}'
echo ""

# 4. Process Information
print_section "4. Process Information"
PIDS=$(pgrep -f "api_server.py" 2>/dev/null)
if [ -n "$PIDS" ]; then
    print_ok "API processes running: $(echo $PIDS | wc -w)"
    echo "PIDs: $PIDS"
    for PID in $PIDS; do
        ps -p $PID -o pid,user,%cpu,%mem,cmd --no-headers
    done
else
    print_error "No API processes found"
fi
echo ""

# 5. Network Status
print_section "5. Network Status"
if netstat -tulpn 2>/dev/null | grep -q ":8080"; then
    print_ok "API is listening on port 8080"
    netstat -tulpn 2>/dev/null | grep ":8080"
else
    print_error "API is not listening on port 8080"
fi
echo ""

# 6. Recent Logs
print_section "6. Recent Logs (Last 10 lines)"
if [ -f /var/log/supervisor/hidrologi-api.out.log ]; then
    echo "Output Log:"
    tail -n 5 /var/log/supervisor/hidrologi-api.out.log
    echo ""
fi

if [ -f /var/log/supervisor/hidrologi-api.err.log ]; then
    echo "Error Log:"
    tail -n 5 /var/log/supervisor/hidrologi-api.err.log
    echo ""
fi

# 7. Storage Status
print_section "7. Storage Status"
RESULTS_DIR="/var/www/itriverdna.my.id/public_html/project_hidrologi_ml/results"
if [ -d "$RESULTS_DIR" ]; then
    JOB_COUNT=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)
    print_ok "Results directory: $JOB_COUNT jobs, $TOTAL_SIZE total"
    
    # Show oldest jobs
    echo "Oldest jobs:"
    find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T+ %p\n' 2>/dev/null | sort | head -3 | while read line; do
        echo "  $line"
    done
else
    print_error "Results directory not found"
fi
echo ""

# 8. Nginx Status
print_section "8. Nginx Status"
if systemctl is-active --quiet nginx; then
    print_ok "Nginx is running"
    systemctl status nginx --no-pager | grep "Active:" | sed 's/^/  /'
else
    print_error "Nginx is not running"
fi
echo ""

# 9. SSL Certificate
print_section "9. SSL Certificate"
if [ -d /etc/letsencrypt/live/itriverdna.my.id ]; then
    CERT_FILE="/etc/letsencrypt/live/itriverdna.my.id/cert.pem"
    if [ -f "$CERT_FILE" ]; then
        EXPIRY=$(openssl x509 -enddate -noout -in "$CERT_FILE" | cut -d= -f2)
        print_ok "SSL Certificate expires: $EXPIRY"
    fi
else
    print_error "SSL Certificate not found"
fi
echo ""

# 10. Recent Activity
print_section "10. Recent Activity (Nginx Access Log)"
if [ -f /var/log/nginx/hidrologi-api-access.log ]; then
    echo "Recent API requests:"
    tail -n 5 /var/log/nginx/hidrologi-api-access.log | awk '{print "  " $0}'
else
    print_info "No access log found"
fi
echo ""

# Footer
echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
echo -e "${GREEN}Monitoring completed at $(date)${NC}"
echo ""
echo "Useful commands:"
echo "  - Restart API: sudo supervisorctl restart hidrologi-api:*"
echo "  - View logs: sudo supervisorctl tail -f hidrologi-api"
echo "  - Follow logs: sudo tail -f /var/log/supervisor/hidrologi-api.out.log"
echo ""
