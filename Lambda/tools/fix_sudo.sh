#!/bin/bash

# Check if the script is run as root
if [ "$(id -u)" -ne 0 ]; then
	  echo "Please run as root or with sudo."
	    exit 1
fi

# Add the current user to the docker group
echo "Adding user $USER to docker group..."
sudo usermod -aG docker $USER

echo "newgrp docker ..."
newgrp docker

# Inform the user to log out and back in
echo "User $USER has been added to the docker group."
echo "Please log out and log back in to apply the changes."

exit 0

