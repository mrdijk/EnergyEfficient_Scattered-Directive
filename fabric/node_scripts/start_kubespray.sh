#!/bin/bash

{
cd kubespray
source ./venv/bin/activate

ansible-playbook -i inventory/dynamos-cluster/inventory.ini cluster.yml -b -v -u ubuntu --private-key=~/.ssh/slice_key
}
