import requests

HEADERS_APPROVAL = {
    "Content-Type": "application/json",
    # Add specific host for this for FABRIC Kubernetes environment
    "Host": "api-gateway.api-gateway.svc.cluster.local"
}

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

NODE_IP = "10.139.1.2"
NODEPORT_BASE_URL = f"http://{NODE_IP}:31141"
APPROVAL_URL = f"{NODEPORT_BASE_URL}/api/v1/requestApproval"

if __name__ == "__main__":
    hfl_request_body = HFL_REQUEST
    headers = HEADERS_APPROVAL.copy()
    requests_url = APPROVAL_URL
    requests.post(requests_url, json=hfl_request_body, headers=headers)
