docker build -f Dockerfile.flwr-api -t simple-flwr-api . 

docker run --rm -t -p 5000:5000 --network flwr-network simple-flwr-api

curl -X POST http://localhost:5000/run-script


curl -X POST http://prets.prets.svc.cluster.local:7000/run-script
curl -X POST http://prets.prets.svc.cluster.local:7000/simple-flwr-api/run-script
curl -X POST http://prets.prets.svc.cluster.local/simple-flwr-api/run-script


