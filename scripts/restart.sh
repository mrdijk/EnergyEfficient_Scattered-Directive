#!/bin/bash

# source /EnergyEfficient_Scattered-Directive/scripts/dynamos-configs.sh
helm uninstall orchestrator
helm uninstall agents
helm uninstall api-gateway
helm uninstall core
kubectl delete ns server $(kubectl get ns --no-headers | awk '/^client[0-9]+/ {print $1}')

echo "Waiting for 60 seconds"
sleep 60

./EnergyEfficient_Scattered-Directive/configuration/dynamos-configuration.sh
