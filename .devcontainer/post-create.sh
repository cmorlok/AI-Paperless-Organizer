#!/bin/bash
set -e

echo "Setting up development environment..."

# Create data directory for SQLite database and ChromaDB
echo "Creating data directories..."
mkdir -p /app/backend/data
sudo chown developer:developer /app/backend/data
echo "Data directory ready at /app/backend/data"

# Install Python packages
if [ -f "/app/backend/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r /app/backend/requirements.txt
fi

# Install Frontend packages
if [ -d "/app/frontend" ]; then
    echo "Installing Frontend dependencies..."
    cd /app/frontend
    npm install
    cd /app
fi

# Install Playwright browsers
if command -v playwright &> /dev/null; then
    echo "Installing Playwright browsers..."
    playwright install chromium
    echo "Playwright browsers installed"
fi

# Set up .vscode/launch.json symlink
echo "Setting up VS Code launch configuration..."
mkdir -p /app/.vscode
if [ -L /app/.vscode/launch.json ]; then
    rm /app/.vscode/launch.json
    echo "   Removed existing symlink"
elif [ -f /app/.vscode/launch.json ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv /app/.vscode/launch.json /app/.vscode/launch.json.bak.$TIMESTAMP
    echo "   Backed up existing launch.json to launch.json.bak.$TIMESTAMP"
fi
ln -s ../.devcontainer/launch.json /app/.vscode/launch.json
echo "Launch configuration symlinked to .vscode/launch.json"

# Set up .vscode/settings.json symlink
echo "Setting up VS Code settings..."
if [ -L /app/.vscode/settings.json ]; then
    rm /app/.vscode/settings.json
    echo "   Removed existing symlink"
elif [ -f /app/.vscode/settings.json ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv /app/.vscode/settings.json /app/.vscode/settings.json.bak.$TIMESTAMP
    echo "   Backed up existing settings.json to settings.json.bak.$TIMESTAMP"
fi
ln -s ../.devcontainer/settings.json /app/.vscode/settings.json
echo "Settings configuration symlinked to .vscode/settings.json"

echo ""
echo "==================================================================="
echo "Development environment setup complete!"
echo ""
echo "Start the backend:"
echo "   cd /app/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Start the frontend:"
echo "   cd /app/frontend && npm run dev"
echo ""
echo "Database Tools:"
echo "   SQLite Viewer extension - Browse DB in VS Code"
echo "   sqlite3 /app/backend/data/organizer.db - Direct CLI access"
echo ""
echo "Database Location: /app/backend/data/organizer.db"
echo "==================================================================="
