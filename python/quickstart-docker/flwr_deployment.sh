#!/bin/bash

# flwr-deployment.sh
# Flower v1.12.0 deployment script based on:
# https://flower.ai/docs/framework/main/fr/docker/tutorial-quickstart-docker.html

# Step 0: Create network (only needed once)
docker network create --driver bridge flwr-network || true

# Step 1: Start Superlink
docker run --rm \
    -p 9091:9091 -p 9092:9092 \
    -v ./data:/app/data \
    --network flwr-network \
    --name superlink \
    --detach \
    flwr/superlink:1.12.0 --insecure

# Step 2: Start Supernodes
docker run --rm -p 9094:9094 \
    --network flwr-network \
    --name supernode-1 \
    --detach \
    -v ./data:/app/data \
    flwr/supernode:1.12.0 \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=0 num-partitions=3" \
    --supernode-address 0.0.0.0:9094 \
    --isolation process

docker run --rm -p 9095:9095 \
    --network flwr-network \
    --name supernode-2 \
    --detach \
    -v ./data:/app/data \
    flwr/supernode:1.12.0 \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=1 num-partitions=3" \
    --supernode-address 0.0.0.0:9095 \
    --isolation process

docker run --rm -p 9096:9096 \
    --network flwr-network \
    --name supernode-3 \
    --detach \
    -v ./data:/app/data \
    flwr/supernode:1.12.0 \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=2 num-partitions=3" \
    --supernode-address 0.0.0.0:9096 \
    --isolation process

# Step 3: Build client app image
docker build -f Dockerfile.clientapp -t flwr_clientapp:0.0.1 .

# NOTE: Ensure model directory has subfolders `central` and `clients`

# Step 4: Start Clients
docker run --rm \
    --network flwr-network \
    --detach \
    -v ./data:/app/data \
    -v ./model:/app/model \
    flwr_clientapp:0.0.1 \
    --supernode supernode-1:9094

docker run --rm \
    --network flwr-network \
    --detach \
    -v ./data:/app/data \
    -v ./model:/app/model \
    flwr_clientapp:0.0.1 \
    --supernode supernode-2:9095

docker run --rm \
    --network flwr-network \
    --detach \
    -v ./data:/app/data \
    -v ./model:/app/model \
    flwr_clientapp:0.0.1 \
    --supernode supernode-3:9096

# Step 5: Build and Start Superexecutor
docker build -f Dockerfile.superexec -t flwr_superexec:0.0.1 .

docker run --rm -p 9093:9093 \
    --network flwr-network \
    --name superexec \
    --detach \
    flwr_superexec:0.0.1 \
    --insecure \
    --executor-config superlink="superlink:9091"

# Note: Ensure pyproject.toml has:
# [tool.flwr.federations.local-deployment]
# address = "127.0.0.1:9093"
# insecure = true
# options.num-supernodes = 3

# Optional Localhost Manual Run:
# flwr run . local-deployment --stream

# Step 6: Build and Run API Trigger
docker build -f Dockerfile.flwr-api-local -t simple-flwr-api .

docker run --rm -p 5000:5000 \
    --network flwr-network \
    simple-flwr-api

# (From another terminal): Trigger via
# curl -X POST http://localhost:5000/run-script

# Step 7: Cleanup (Run this when done)
# docker stop $(docker ps -a -q --filter ancestor=flwr_clientapp:0.0.1) \
#     supernode-1 supernode-2 supernode-3 superexec superlink
