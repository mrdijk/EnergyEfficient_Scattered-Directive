based on https://flower.ai/docs/framework/main/fr/docker/tutorial-quickstart-docker.html 
with some adjustments 
--------------

docker network create --driver bridge flwr-network

docker run --rm \
      -p 9091:9091 -p 9092:9092 \
	  -v ./data:/app/data \
      --network flwr-network \
      --name superlink \
      --detach \
      flwr/superlink:1.12.0 --insecure
	  
docker run --rm \
    -p 9094:9094 \
    --network flwr-network \
    --name supernode-1 \
    --detach \
	-v ./data:/app/data \
    flwr/supernode:1.12.0  \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=0 num-partitions=3" \
    --supernode-address 0.0.0.0:9094 \
    --isolation process
	

docker run --rm \
    -p 9095:9095 \
    --network flwr-network \
    --name supernode-2 \
    --detach \
	-v ./data:/app/data \
    flwr/supernode:1.12.0  \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=1 num-partitions=3" \
    --supernode-address 0.0.0.0:9095 \
    --isolation process


docker run --rm \
    -p 9096:9096 \
    --network flwr-network \
    --name supernode-3 \
    --detach \
	-v ./data:/app/data \
    flwr/supernode:1.12.0  \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=2 num-partitions=3" \
    --supernode-address 0.0.0.0:9096 \
    --isolation process

docker build -f Dockerfile.clientapp -t flwr_clientapp:0.0.1 .

docker run --rm \
    --network flwr-network \
    --detach \
	-v ./data:/app/data \
	-v ./model:/app/model \
    flwr_clientapp:0.0.1  \
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

----ALTERNATIVELY (WITHOUT MOUNTED VOLUMES)

docker run --rm \
    --network flwr-network \
    --detach \
    flwr_clientapp:0.0.1  \
    --supernode supernode-1:9094

docker run --rm \
    --network flwr-network \
    --detach \
    flwr_clientapp:0.0.1 \
    --supernode supernode-2:9095

docker run --rm \
    --network flwr-network \
    --detach \
    flwr_clientapp:0.0.1 \
    --supernode supernode-3:9096

---------

docker build -f Dockerfile.superexec -t flwr_superexec:0.0.1 .

docker run --rm \
   -p 9093:9093 \
    --network flwr-network \
    --name superexec \
    --detach \
    flwr_superexec:0.0.1 \
    --insecure \
    --executor-config superlink=\"superlink:9091\"	

(APPEND TO pyproject.toml) 
[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"
insecure = true
options.num-supernodes = 3

--------

flwr run . local-deployment --stream

--
OR run via the API trigger (see below)


----------------
DOCKER flwr trigger 

# docker build -t simple-flwr-api . 
# docker build -f Dockerfile.flwr-api -t simple-flwr-api .
docker build -f Dockerfile.flwr-api-local -t simple-flwr-api .

docker run --rm -p 5000:5000 --network flwr-network simple-flwr-api


--
(on different terminal) run: 

curl -X POST http://localhost:5000/run-script

--------------
REMOVE DOCKER CONTAINERS
(clean-up)

docker stop $(docker ps -a -q  --filter ancestor=flwr_clientapp:0.0.1) \
   supernode-1 \
   supernode-2 \
   supernode-3 \
   superexec \
   superlink