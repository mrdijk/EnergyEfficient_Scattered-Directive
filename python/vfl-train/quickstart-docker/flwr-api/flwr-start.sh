echo "Starting flwr deployment from docker container"
flwr run . local-deployment --stream --run-config 'address = "127.0.0.1:9093"'
