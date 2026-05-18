# FABRIC specific variables:
# NodePort address and exposed port of the API Gateway service (see testing example with curl in fabric/dynamos/DYNAMOS_setup.ipynb notebook)
# Can be fetched with the following command to extract INTERNAL-IP of the node: kubectl get nodes -o wide
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODE_IP = "10.145.1.3"
# Replace with corresponding node IP and NodePort if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc -n ingress -n ingress
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODEPORT_BASE_URL = f"http://{NODE_IP}:32526"

# Prometheus. The URL is specific to the FABRIC Kubernetes environment, so this should be changed if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
PROMETHEUS_URL = f"http://{NODE_IP}:30791"
PROM_CONTAINER_NS = '{container_namespace=~"kernel_processes|system_processes|client.*|server.*|policy.*|orchestrator|sidecar|rabbitmq|api-gateway"}'
PROM_KEPLER_ENERGY_METRIC = "kepler_container_joules_total"
PROM_KEPLER_CONTAINER_LABEL = "container_namespace, pod_name, container_name"
PROM_ENERGY_QUERY_TOTAL = f"sum({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINER_NS}) by ({PROM_KEPLER_CONTAINER_LABEL})"
PROM_ENERGY_QUERY_RANGE = f"sum(increase({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINER_NS}[2m])) by ({PROM_KEPLER_CONTAINER_LABEL})"

# Add specific FABRIC Kubernetes setup for these urls
APPROVAL_URL = f"{NODEPORT_BASE_URL}/api/v1/requestApproval"
HEADERS = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "api-gateway.api-gateway.svc.cluster.local",
}

# URL for polling the status of the current training
STATUS_URL = f"http://{NODE_IP}:32526/api/v1/getTrainingStatus?id="

# HFL request body
HFL_REQUEST = {
    "type": "hflTrainModelRequest",
    "user": {"id": "1234", "userName": "maurits.dijk@student.uva.nl"},
    "data_request": {"type": "hflTrainModelRequest", "data": {}, "requestMetadata": {}},
}
