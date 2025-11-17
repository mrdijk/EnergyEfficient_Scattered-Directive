#!/bin/bash

source ./EnergyEfficient_Scattered-Directive/configuration/dynamos-configs.sh
uninstall_all
kubectl delete ns clientone clienttwo clientthree server

sleep 60

./EnergyEfficient_Scattered-Directive/configuration/dynamos-configuration.sh
