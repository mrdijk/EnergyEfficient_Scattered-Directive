#!/bin/bash

{
git clone https://github.com/mrdijk/EnergyEfficient_Scattered-Directive.git

sudo mkdir -p /mnt/etcd-data
sudo cp ~/DYEnergyEfficient_Scattered-DirectiveNAMOS/configuration/etcd_launch_files/*.json /mnt/etcd-data
sudo chmod -R 777 /mnt/etcd-data
}
