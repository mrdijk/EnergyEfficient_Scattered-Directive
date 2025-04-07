#!/bin/bash

{
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

git clone https://github.com/kubernetes-sigs/kubespray.git
cd kubespray
git checkout release-2.27
rm -rf .git

python3 -m venv venv
source ./venv/bin/activate

pip3 install -r requirements.txt

mv inventory/sample inventory/dynamos

ansible --version
}  2>&1 | tee -a start_control_plane.log
