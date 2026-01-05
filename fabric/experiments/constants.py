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
# PROM_CONTAINERS = "{container_name=~\"kernel_processes|system_processes|client(one|two|three)|server|sql.*|policy.*|orchestrator|sidecar|rabbitmq|api-gateway\"}"
PROM_CONTAINER_NS = "{container_namespace=~\"kernel_processes|system_processes|client([1-9]|1[0-9]|20)|server|policy.*|orchestrator|sidecar|rabbitmq|api-gateway\"}"
PROM_KEPLER_ENERGY_METRIC = "kepler_container_joules_total"
PROM_KEPLER_CONTAINER_LABEL = ['container_namespace', 'pod_name', 'container_name']
PROM_ENERGY_QUERY_TOTAL = f"sum({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINER_NS}) by ({PROM_KEPLER_CONTAINER_LABEL})"
PROM_ENERGY_QUERY_RANGE = f"sum(increase({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINER_NS}[2m])) by ({PROM_KEPLER_CONTAINER_LABEL})"

# Experiment configurations
IDLE_PERIOD = 120  # Idle period in seconds
ACTIVE_PERIOD = 120  # Active period in seconds
LEARNING_RATE = 0.01
DATA_PROVIDERS =  {'client1': 3799, 'client2': 10570, 'client3': 4725, 'client4': 2182, 'client5': 17938, 
									  'client6': 2447, 'client7': 1681, 'client8': 1729, 'client9': 6896, 'client10': 14812, 
									   'client12': 3746, 'client13': 4337, 'client14': 2146, 
									  'client16': 1711, 'client17': 2094, 'client18': 3188, 'client20': 8281}
# Removed 'client11': 2778, 'client15': 2665 and 'client19': 2265  due to resource constraints (too little disk space)
# clients are supposed to have 100GB, client15 has 10GB
# Mean number of rows is 5428 instead of 5000

# Add specific FABRIC Kubernetes setup for these urls
APPROVAL_URL = f"{NODEPORT_BASE_URL}/api/v1/requestApproval"
HEADERS = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "api-gateway.api-gateway.svc.cluster.local"
}
 
# URL for polling the status of the current training
STATUS_URL = f'http://{NODE_IP}:32526/api/v1/getTrainingStatus?id='

#HFL request body
HFL_REQUEST = {
    "type": "hflTrainModelRequest",
    "user": {
        "id": "1234",
        "userName": "maurits.dijk@student.uva.nl"
    },
    "data_request": {
        "type": "hflTrainModelRequest",
        "data": {},
        "requestMetadata": {}
    }
}
