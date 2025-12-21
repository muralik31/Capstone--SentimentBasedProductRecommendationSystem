#!/usr/bin/env bash
# Build script for Render deployment

set -e

# Install git-lfs and pull large files
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install
git lfs pull

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"

