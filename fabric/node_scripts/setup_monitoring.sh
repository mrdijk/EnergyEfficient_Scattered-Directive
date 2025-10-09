#!/bin/bash

# This script is used to setup monitoring in DYNAMOS. See energy-efficiency/docs/getting-started/3_SetupMonitoring.md for more explanation.
# This is just a copy of the scripts provided there, but then slightly modified specifically for Kubernetes in FABRIC, where indicated with a comment "FABRIC EDIT".

# ================================================ 0: Setup, present in both below scripts ================================================
# Change this to the path of the DYNAMOS repository on your disk
echo "Setting up paths..."
DYNAMOS_ROOT="${HOME}/DYNAMOS"
# Charts
charts_path="${DYNAMOS_ROOT}/charts"
monitoring_chart="${charts_path}/monitoring"
monitoring_values="$monitoring_chart/values.yaml"

# ================================================ 1: energy-efficiency/scripts/prepare-monitoring/prometheusAndGrafana.sh ================================================
# Create the namespace in the Kubernetes cluster (if not exists)
kubectl create namespace monitoring

# Install and add Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Additional information to helm chart used: https://artifacthub.io/packages/helm/prometheus-community/kube-prometheus-stack
# It is a widely used chart and maintained extensively (at least this was at the time of creating: 2025). Formerly named: prometheus-operator

# Install prometheus stack (this may take a while before the pods are running (sometimes even up to minutes))
# -i flag allows helm to install it if it does not exist yet, otherwise upgrade it
# Use the monitoring namcespace for prometheus (and use config file with the -f flag)
# Using upgrade ensures that helm manages it correctly, this will upgrade or install if not exists
# This names the release 'prometheus'. This is VERY IMPORTANT, because this release will be used by Kepler and others to create ServiceMonitors for example
# Use specific version to ensure compatability (this version has worked in previous setups)
helm upgrade -i prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --version 68.1.0 \
    -f "$monitoring_chart/prometheus-config.yaml"
# Prometheus stack already includes grafana itself with a default setup (saves time to set it up yourself)
# Uninstall the release using helm to rollback changes: helm uninstall prometheus --namespace monitoring


# FABRIC EDIT: waiting 30 seconds to make sure everything is running before running next script
sleep 60
# This result has to be all running before the next part can be executed
kubectl --namespace monitoring get pods -l "release=prometheus"


# ================================================ 2: energy-efficiency/scripts/prepare-monitoring/keplerAndMonitoringChart.sh ================================================
# More information on Kepler installation: https://sustainable-computing.io/installation/kepler-helm/
# Installing Prometheus (and Grafana) can be skipped, this is already done earlier 

# Install and add Kepler
helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
helm repo update
# Install Kepler
# This also creates a service monitor for the prometheus stack
# Use specific version to ensure compatability (this version has worked in previous setups)
# FABRIC EDIT: do NOT set nodeSelector here, since it will then only add it to that node, but this needs to run on every node.
helm upgrade -i kepler kepler/kepler \
    --namespace monitoring \
    --version 0.5.12 \
    --set serviceMonitor.enabled=true \
    --set serviceMonitor.labels.release=prometheus \
# Uninstall the release using helm to rollback changes: helm uninstall kepler -n monitoring

# Apply/install the monitoring helm release (will use the monitoring charts,
# which includes the deamonset, service and sesrvicemonitor for cadvisor for example)
# Optional: enable debug flag to output logs for potential errors (add --debug to the end of the next line)
# FABRIC EDIT: this uses the corresponding node by specifying it in the charts folder.
helm upgrade -i monitoring $monitoring_chart --namespace monitoring -f "$monitoring_values"
# Uninstall the release using helm to rollback changes: helm uninstall monitoring --namespace monitoring
