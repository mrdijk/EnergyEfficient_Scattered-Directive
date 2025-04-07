#!/bin/bash

subnet=$1
ip=$2

{
git clone https://github.com/kubernetes-sigs/kubespray.git
cd kubespray
git checkout release-2.27
pip3 install -r requirements.txt

kubectl get nodes

}  2>&1 | tee -a start_control_plane.log
