echo "Starting flwr deployment from docker container (for DYNAMOS)"
flwr run . local-deployment --stream --run-config 'address = "0.0.0.0:9093"'
