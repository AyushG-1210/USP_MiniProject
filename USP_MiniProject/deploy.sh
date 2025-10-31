#!/bin/bash
set -e # This makes the script exit if any command fails

echo "Starting deployment..."

# 1. Pull the latest code from the main branch
git pull origin main

# 2. Build the new Docker image (with sudo)
sudo docker build -t myapp:latest .

# 3. Stop the old container (with sudo)
sudo docker stop myapp || true

# 4. Run the new container (with sudo)
sudo docker run -d --rm --name myapp -p 8080:8080 myapp:latest

echo "Deployment successful!"