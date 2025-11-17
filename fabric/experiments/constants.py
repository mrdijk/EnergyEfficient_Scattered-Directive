# FABRIC specific variables:
# NodePort address and exposed port of the API Gateway service (see testing example with curl in fabric/dynamos/DYNAMOS_setup.ipynb notebook)
# Can be fetched with the following command to extract INTERNAL-IP of the node: kubectl get nodes -o wide
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODE_IP = "10.139.1.2"
# Replace with corresponding node IP and NodePort if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc -n ingress -n ingress
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODEPORT_BASE_URL = f"http://{NODE_IP}:31141"

# Prometheus. The URL is specific to the FABRIC Kubernetes environment, so this should be changed if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
PROMETHEUS_URL = f"http://{NODE_IP}:32535"
PROM_CONTAINERS = "{container_name=~\"kernel_processes|system_processes|client(one|two|three)|server|sql.*|policy.*|orchestrator|sidecar|rabbitmq|api-gateway\"}"
PROM_KEPLER_ENERGY_METRIC = "kepler_container_joules_total"
PROM_KEPLER_CONTAINER_LABEL = "container_name"
PROM_ENERGY_QUERY_TOTAL = f"sum({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINERS}) by ({PROM_KEPLER_CONTAINER_LABEL})"
PROM_ENERGY_QUERY_RANGE = f"sum(increase({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINERS}[3m])) by ({PROM_KEPLER_CONTAINER_LABEL})"

# Experiment configurations
IDLE_PERIOD = 120  # Idle period in seconds
ACTIVE_PERIOD = 180  # Active period in seconds
ROUNDS = 10 # Defaults number of training rounds
DATA_PROVIDERS =  ["server", "clientone", "clienttwo", "clientthree"]

# Add specific FABRIC Kubernetes setup for these urls
APPROVAL_URL = f"{NODEPORT_BASE_URL}/api/v1/requestApproval"
HEADERS_APPROVAL = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "api-gateway.api-gateway.svc.cluster.local"
}
 
#HFL request body
HFL_REQUEST = {
    "type": "hflTrainModelRequest",
    "user": {
        "id": "1234",
        "userName": "maurits.dijk@student.uva.nl@student.uva.nl"
    },
    "dataProviders": ["server", "clientone", "clienttwo", "clientthree"],
    "data_request": {
        "type": "hflTrainModelRequest",
        "data": {},
        "requestMetadata": {}
    }
}