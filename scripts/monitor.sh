#!/bin/bash

#############################################
# HIDROLOGI ML - Monitoring Script
# Real-time monitoring untuk aplikasi
#############################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_USER="hidrologi"
APP_DIR="/home/$APP_USER/project_hidrologi_ml"
SERVICE_NAME="hidrologi-api.service"

clear
echo -e "${BLUE}"
echo "================================================"
echo "  HIDROLOGI ML - System Monitor"
echo "================================================"
echo -e "${NC}"

# Function to get service status
get_service_status() {
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}●${NC} Running"
    else
        echo -e "${RED}●${NC} Stopped"
    fi
}

# Function to get CPU usage
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
}

# Function to get memory usage
get_memory_usage() {
    free -h | awk '/^Mem:/ {printf "%s / %s (%.1f%%)", $3, $2, $3/$2*100}'
}

# Function to get disk usage
get_disk_usage() {
    df -h $APP_DIR | awk 'NR==2 {printf "%s / %s (%s)", $3, $2, $5}'
}

# Function to get active jobs
get_active_jobs() {
    if [ -d "$APP_DIR/results" ]; then
        find $APP_DIR/results -maxdepth 1 -type d | wc -l
    else
        echo "0"
    fi
}

# Function to get recent errors
get_recent_errors() {
    if [ -f "$APP_DIR/logs/api_error.log" ]; then
        tail -5 $APP_DIR/logs/api_error.log | grep -c "ERROR" || echo "0"
    else
        echo "0"
    fi
}

# Main monitoring loop
while true; do
    # Get current time
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get metrics
    SERVICE_STATUS=$(get_service_status)
    CPU_USAGE=$(get_cpu_usage)
    MEMORY_USAGE=$(get_memory_usage)
    DISK_USAGE=$(get_disk_usage)
    ACTIVE_JOBS=$(get_active_jobs)
    RECENT_ERRORS=$(get_recent_errors)
    
    # Clear screen and display dashboard
    clear
    echo -e "${BLUE}"
    echo "================================================"
    echo "  HIDROLOGI ML - System Monitor"
    echo "================================================"
    echo -e "${NC}"
    echo -e "${YELLOW}Last Updated:${NC} $CURRENT_TIME"
    echo ""
    
    # Service Status
    echo -e "${BLUE}Service Status:${NC}"
    echo -e "  Status: $SERVICE_STATUS"
    echo -e "  Uptime: $(systemctl show $SERVICE_NAME --property=ActiveEnterTimestamp --value 2>/dev/null || echo 'N/A')"
    echo ""
    
    # System Resources
    echo -e "${BLUE}System Resources:${NC}"
    echo -e "  CPU Usage:    $CPU_USAGE"
    echo -e "  Memory Usage: $MEMORY_USAGE"
    echo -e "  Disk Usage:   $DISK_USAGE"
    echo ""
    
    # Application Metrics
    echo -e "${BLUE}Application Metrics:${NC}"
    echo -e "  Active Jobs:    $ACTIVE_JOBS"
    echo -e "  Recent Errors:  $RECENT_ERRORS"
    echo ""
    
    # Recent Logs
    echo -e "${BLUE}Recent Logs (last 5 lines):${NC}"
    if [ -f "$APP_DIR/logs/api.log" ]; then
        tail -5 $APP_DIR/logs/api.log | sed 's/^/  /'
    else
        echo "  No logs available"
    fi
    echo ""
    
    # Quick Actions
    echo -e "${YELLOW}Quick Actions:${NC}"
    echo "  [R] Restart Service"
    echo "  [L] View Full Logs"
    echo "  [S] Service Status Detail"
    echo "  [Q] Quit"
    echo ""
    
    # Wait for input with timeout
    read -t 5 -n 1 action
    
    case $action in
        r|R)
            echo ""
            echo -e "${YELLOW}Restarting service...${NC}"
            sudo systemctl restart $SERVICE_NAME
            sleep 2
            ;;
        l|L)
            echo ""
            echo -e "${YELLOW}Opening logs (Ctrl+C to return)...${NC}"
            sleep 1
            tail -f $APP_DIR/logs/api.log
            ;;
        s|S)
            echo ""
            echo -e "${YELLOW}Service status:${NC}"
            sudo systemctl status $SERVICE_NAME --no-pager
            echo ""
            read -p "Press Enter to continue..."
            ;;
        q|Q)
            echo ""
            echo -e "${GREEN}Exiting monitor...${NC}"
            exit 0
            ;;
    esac
done
