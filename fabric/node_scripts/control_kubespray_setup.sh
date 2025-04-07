#!/bin/bash

{
git clone https://github.com/kubernetes-sigs/kubespray.git
cd kubespray
git checkout release-2.27
pip3 install -r requirements.txt

mv inventory/sample inventory/dynamos

ansible version
}  2>&1 | tee -a start_control_plane.log
