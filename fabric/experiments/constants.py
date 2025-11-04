# FABRIC specific variables:
# NodePort address and exposed port of the API Gateway service (see testing example with curl in fabric/dynamos/DYNAMOS_setup.ipynb notebook)
# Can be fetched with the following command to extract INTERNAL-IP of the node: kubectl get nodes -o wide
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODE_IP = "10.145.6.5"
# Replace with corresponding node IP and NodePort if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc -n ingress -n ingress
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
NODEPORT_BASE_URL = f"http://{NODE_IP}:30635"
API_GATEWAY_POD = "api-gateway-7b6949b88c-87cjv"

# Experiment script values
# All prefixes, i.e. implementations
IMPLEMENTATIONS_PREFIXES = ["baseline", "compression", "caching"]
OPTIMIZATIONS_PREFIXES = ["compression", "caching"]
ARCHETYPES = ["ComputeToData", "DataThroughTTP"]
ARCHETYPE_ACRONYMS = {
    "ComputeToData": "CtD", 
    "DataThroughTTP": "DtTTP"
}

# Prometheus. The URL is specific to the FABRIC Kubernetes environment, so this should be changed if Kubernetes/FABRIC nodes have been reconfigured/recreated
# Can be fetched with the following command to extract the NodePort from <LocalNodePort>:<NodePort>/TCP with: kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring
# See fabric/dynamos/DYNAMOS_setup.ipynb notebook for an example and more explanation on this.
PROMETHEUS_URL = f"http://{NODE_IP}:31483"
PROM_CONTAINERS = "{container_name=~\"kernel_processes|system_processes|client(one|two|three)|server|sql.*|policy.*|orchestrator|sidecar|rabbitmq|api-gateway\"}"
PROM_KEPLER_ENERGY_METRIC = "kepler_container_joules_total"
PROM_KEPLER_CONTAINER_LABEL = "container_name"
PROM_ENERGY_QUERY_TOTAL = f"sum({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINERS}) by ({PROM_KEPLER_CONTAINER_LABEL})"
PROM_ENERGY_QUERY_RANGE = f"sum(increase({PROM_KEPLER_ENERGY_METRIC}{PROM_CONTAINERS}[2m])) by ({PROM_KEPLER_CONTAINER_LABEL})"

# Experiment configurations
NUM_EXP_ACTIONS = 7  # Number of actions per experiment
IDLE_PERIOD = 120  # Idle period in seconds
ACTIVE_PERIOD = 120  # Active period in seconds

# DYNAMOS requests
# Add specific FABRIC Kubernetes setup for these urls
REQUEST_URLS = {
    "uva": f"{NODEPORT_BASE_URL}/agent/v1/sqlDataRequest/uva",
    "surf": f"{NODEPORT_BASE_URL}/agent/v1/sqlDataRequest/surf"
}
HEADERS = {
    "Content-Type": "application/json",
    # Access token required for data requests in DYNAMOS
    "Authorization": "bearer 1234"
}
INITIAL_REQUEST_BODY = {
    "type": "sqlDataRequest",
    "query": "SELECT DISTINCT p.Unieknr, p.Geslacht, p.Gebdat, s.Aanst_22, s.Functcat, s.Salschal as Salary FROM Personen p JOIN Aanstellingen s ON p.Unieknr = s.Unieknr LIMIT 30000",
    "algorithm": "",
    "options": {"graph": False, "aggregate": False},
    "user": {"id": "12324", "userName": "jorrit.stutterheim@cloudnation.nl"},
}
# Add specific FABRIC Kubernetes setup for these urls
APPROVAL_URL = f"{NODEPORT_BASE_URL}/api/v1/requestApproval"
HEADERS_APPROVAL = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "api-gateway.api-gateway.svc.cluster.local"
}
REQUEST_BODY_APPROVAL = {
    "type": "sqlDataRequest",
    "user": {
        "id": "12324",
        "userName": "jorrit.stutterheim@cloudnation.nl"
    },
    "dataProviders": ["UVA"]
}

#VFL request
VFL_REQUEST = {
    "type": "vflTrainModelRequest",
    "user": {
        "id": "1234",
        "userName": "jake.jongejans@student.uva.nl"
    },
    "dataProviders": ["server", "clientone", "clienttwo", "clientthree"],
    "data_request": {
        "type": "vflTrainModelRequest",
        "data": {
            "learning_rate": 0.05,
            "cycles": 10,
            "change_policies": 10
        },
        "requestMetadata": {}
    }
}

# Update archetypes
# Add specific FABRIC Kubernetes setup for these urls
UPDATE_ARCH_URL = f"{NODEPORT_BASE_URL}/api/v1/archetypes/agreements"
INITIAL_REQUEST_BODY_ARCH = {
    "name": "computeToData",
    "computeProvider": "dataProvider",
    "resultRecipient": "requestor",
}
HEADERS_UPDATE_ARCH = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "orchestrator.orchestrator.svc.cluster.local"
}
WEIGHTS = {
    "ComputeToData": 100,
    "DataThroughTTP": 300
}
ARCH_DATA_STEWARDS = {
    # Each archetype has a different data steward it should request the data from
    "ComputeToData": "uva",
    "DataThroughTTP": "surf"
}