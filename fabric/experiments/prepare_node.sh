#!/bin/bash

# This script prepares the node by installing the requirements in a virtual python environment where the experiments can be executed

# Update apt, use noninteractive mode (this is where the script is executed in Jupyter Notebook), and use queit mode (-qq)
sudo DEBIAN_FRONTEND=noninteractive apt update -qq && sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y -qq
# Install python, venv and pip
sudo DEBIAN_FRONTEND=noninteractive apt install python3 python3-venv python3-pip -y
# Verify installations:
python3 --version
pip3 --version

# Create virtual python environment in the experiments folder
cd ~/experiments
python3 -m venv dynamos-env
source dynamos-env/bin/activate

# Upgrade pip and install requirements:
# Ensure pip, setuptools, and wheel are up to date (will otherwise fail installation for requirements)
pip install --upgrade pip wheel setuptools
# Install requirements manually, only the ones needed to execute the experiments
pip install numpy requests
# Now the preparation is done and the next steps can be performed.