#!/bin/bash

# Script to install udev rules for ELRS receiver (CP2102 USB-TTL).

set -e

# Check if the script is run as root
if [[ "$EUID" -ne 0 ]]; then
  echo "❌ Please run this script as root (e.g., with sudo)"
  exit 1
fi

echo "🔧 Creating udev rules in /etc/udev/rules.d/"

# Create 99-elrs.rules
# CP2102 (Silicon Labs): idVendor=10c4, idProduct=ea60
cat <<EOF > /etc/udev/rules.d/99-elrs.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="ELRS", MODE="0666"
EOF
echo "✅ Created 99-elrs.rules"

# Reload and apply the new rules
echo "🔄 Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "🎉 ELRS udev rules installed and applied successfully!"
echo "ℹ️  ELRS receiver will be available at /dev/ELRS"
