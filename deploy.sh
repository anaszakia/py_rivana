#!/bin/bash

# ============================================
# Deployment Script - Hidrologi ML API
# ============================================

set -e  # Exit on error

echo "🚀 Starting deployment process..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/var/www/itriverdna.my.id/public_html/project_hidrologi_ml"
VENV_DIR="$PROJECT_DIR/venv"
USER="www-data"

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root or with sudo"
    exit 1
fi

# Step 1: Pull latest code
print_message "Pulling latest code from GitHub..."
cd "$PROJECT_DIR"
sudo -u $USER git pull origin main
if [ $? -eq 0 ]; then
    print_message "✓ Code updated successfully"
else
    print_error "Failed to pull code"
    exit 1
fi

# Step 2: Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    print_warning "Virtual environment not found. Creating..."
    sudo -u $USER python3.11 -m venv "$VENV_DIR"
    print_message "✓ Virtual environment created"
fi

# Step 3: Update dependencies
print_message "Installing/updating Python dependencies..."
sudo -u $USER bash -c "source $VENV_DIR/bin/activate && pip install --upgrade pip"
sudo -u $USER bash -c "source $VENV_DIR/bin/activate && pip install -r requirements.txt"
if [ $? -eq 0 ]; then
    print_message "✓ Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Step 4: Set proper permissions
print_message "Setting proper permissions..."
chown -R $USER:$USER "$PROJECT_DIR"
chmod -R 755 "$PROJECT_DIR"
chmod -R 775 "$PROJECT_DIR/results"
if [ -f "$PROJECT_DIR/gee-credentials.json" ]; then
    chmod 600 "$PROJECT_DIR/gee-credentials.json"
    chown $USER:$USER "$PROJECT_DIR/gee-credentials.json"
fi
print_message "✓ Permissions set"

# Step 5: Run database migrations (if any)
# Uncomment if you have migrations
# print_message "Running database migrations..."
# sudo -u $USER bash -c "source $VENV_DIR/bin/activate && python manage.py migrate"

# Step 6: Restart services
print_message "Restarting API service..."
supervisorctl restart hidrologi-api:*
if [ $? -eq 0 ]; then
    print_message "✓ API service restarted"
else
    print_error "Failed to restart API service"
    exit 1
fi

# Step 7: Reload nginx
print_message "Reloading Nginx..."
systemctl reload nginx
if [ $? -eq 0 ]; then
    print_message "✓ Nginx reloaded"
else
    print_warning "Failed to reload Nginx (continuing anyway)"
fi

# Step 8: Check service status
print_message "Checking service status..."
sleep 3
supervisorctl status hidrologi-api:*

# Step 9: Test API
print_message "Testing API endpoint..."
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8080/health)
if [ "$HTTP_CODE" == "200" ]; then
    print_message "✓ API is responding (HTTP $HTTP_CODE)"
else
    print_error "API is not responding correctly (HTTP $HTTP_CODE)"
    print_message "Check logs: sudo supervisorctl tail -f hidrologi-api"
fi

# Step 10: Clean old results (older than 30 days)
print_message "Cleaning old results..."
find "$PROJECT_DIR/results/"* -mtime +30 -type d -exec rm -rf {} + 2>/dev/null
print_message "✓ Old results cleaned"

# Done
echo ""
print_message "=================================="
print_message "✅ Deployment completed successfully!"
print_message "=================================="
echo ""
print_message "Useful commands:"
echo "  - Check logs: sudo supervisorctl tail -f hidrologi-api"
echo "  - View status: sudo supervisorctl status"
echo "  - Restart: sudo supervisorctl restart hidrologi-api:*"
echo ""
print_message "API Endpoint: https://itriverdna.my.id/api/"
echo ""
