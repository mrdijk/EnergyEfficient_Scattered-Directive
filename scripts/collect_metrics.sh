#!/bin/bash

DYNAMOS_PORT=$(kubectl get svc -n ingress | grep "nginx-nginx-ingress-controller" | sed "s/.*80:\([0-9]*\)\/TCP.*/\1/")
DYNAMOS_IP=$(kubectl get nodes -o wide | grep dynamos | sed "s/.*\s\([0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\).*/\1/")

for number in 5 10 25 50 100; do
    for ((i=1; i<6; i++)); do
        ./configuration/dynamos-configuration.sh

        # Sleep so the API gateway actually gets a message.
        sleep 15

        echo "Running for $number rounds for the ${i}th time."
        curl -H "Host: api-gateway.api-gateway.svc.cluster.local" http://$DYNAMOS_IP:$DYNAMOS_PORT/api/v1/requestApproval \
        --header 'Content-Type: application/json' \
        --data-raw "{
            \"type\": \"vflTrainModelRequest\",
            \"user\": {
                \"id\": \"1234\",
                \"userName\": \"jake.jongejans@student.uva.nl\"
            },
            \"dataProviders\": [\"server\", \"clientone\", \"clienttwo\", \"clientthree\"],
            \"data_request\": {
                \"type\": \"vflTrainModelRequest\",
                \"data\": {
                  \"learning_rate\": 0.05,
                  \"cycles\": $number
                },
        \"requestMetadata\": {}
            }
        }"

        # Wait until training is done
        sleep $number * 12

        ./scripts/retrieve_data.sh "$number-$i"

        sleep 5

        helm uninstall agents api-gateway core orchestrator namespaces prometheus thirdparties

        sleep 30
    done
done
