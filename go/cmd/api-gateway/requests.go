// This file contains the handlers for the requests that the API Gateway receives from the client
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Jorrit05/DYNAMOS/pkg/api"
	"github.com/Jorrit05/DYNAMOS/pkg/lib"
	pb "github.com/Jorrit05/DYNAMOS/pkg/proto"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opencensus.io/trace"
)

func requestHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger.Debug("Starting requestApprovalHandler")
		ctxWithTimeout, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()

		// Start a new span with the context that has a timeout
		ctx, span := trace.StartSpan(ctxWithTimeout, "requestApprovalHandler")
		defer span.End()

		body, err := api.GetRequestBody(w, r, serviceName)
		if err != nil {
			return
		}

		var apiReqApproval api.RequestApproval
		if err := json.Unmarshal(body, &apiReqApproval); err != nil {
			logger.Sugar().Errorf("Error unmMarshalling get apiReqApproval: %v", err)
			return
		}

		userPb := &pb.User{
			Id:       apiReqApproval.User.Id,
			UserName: apiReqApproval.User.UserName,
		}

		var dataRequestInterface map[string]any
		if err := json.Unmarshal(apiReqApproval.DataRequest, &dataRequestInterface); err != nil {
			logger.Sugar().Errorf("Error unmarhsalling get request: %v", err)
			return
		}

		dataRequestOptions := &api.DataRequestOptions{}
		dataRequestOptions.Options = make(map[string]bool)
		if err := json.Unmarshal(apiReqApproval.DataRequest, &dataRequestOptions); err != nil {
			logger.Sugar().Errorf("Error unmMarshalling get apiReqApproval: %v", err)
			return
		}

		dataRequestInterface["user"] = userPb

		// Create protobuf struct for the req approval flow
		protoRequest := &pb.RequestApproval{
			Type:             apiReqApproval.Type,
			User:             userPb,
			DataProviders:    apiReqApproval.DataProviders,
			DestinationQueue: "policyEnforcer-in",
			Options:          dataRequestOptions.Options,
		}

		// Create a channel to receive the response
		responseChan := make(chan validation)

		requestApprovalMutex.Lock()
		requestApprovalMap[protoRequest.User.Id] = responseChan
		requestApprovalMutex.Unlock()

		_, err = c.SendRequestApproval(ctx, protoRequest)
		if err != nil {
			logger.Sugar().Errorf("error in sending requestapproval: %v", err)
		}

		select {
		case validationStruct := <-responseChan:
			msg := validationStruct.response

			logger.Sugar().Infof("Received response, %s", msg.Type)
			if msg.Type != "requestApprovalResponse" {
				logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
				return
			}

			requestMetadata := &pb.RequestMetadata{
				JobId: msg.JobId,
			}
			dataRequestInterface["requestMetadata"] = requestMetadata

			logger.Sugar().Infof("Data Prepared jsonData: %s", dataRequestInterface)

			var response []byte

			if apiReqApproval.Type == "vflTrainModelRequest" {
				response = runVFLTraining(dataRequestInterface, msg.AuthorizedProviders, msg.JobId)
			} else {
				// Marshal the combined data back into JSON for forwarding
				dataRequestJson, err := json.Marshal(dataRequestInterface)
				if err != nil {
					logger.Sugar().Errorf("Error marshalling combined data: %v", err)
					return
				}

				response = sendDataToAuthProviders(dataRequestJson, msg.AuthorizedProviders, apiReqApproval.Type, msg.JobId)
			}

			w.WriteHeader(http.StatusOK)
			w.Write(response)
			return

		case <-ctx.Done():
			http.Error(w, "Request timed out", http.StatusRequestTimeout)
			return
		}
	}
}

// This runs a single round of traingin
func runVFLTrainingRound(dataRequest map[string]any, clients map[string]string, serverAuth string, serverUrl string) ([]string, error) {
	var wg sync.WaitGroup
	var responses []string

	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		dataRequest["type"] = "vflTrainRequest"
		logger.Sugar().Info("This is the dataRequest: ", dataRequest)

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return []string{}, err
		}

		// TODO: Remove unnecessary data in this request
		logger.Sugar().Infof("Sending request to %s.\nEndpoint: %s\nJSON:%v", target, endpoint, string(dataRequestJson))

		go func() {
			responseData, err := sendData(endpoint, dataRequestJson)

			if err != nil {
				logger.Sugar().Errorf("Error sending data, %v", err)
			} else {
				logger.Sugar().Info("responseData: ", responseData)

				responseJson := &pb.MicroserviceCommunication{}
				err = json.Unmarshal([]byte(responseData), responseJson)

				if err != nil {
					logger.Sugar().Error("Unmarshalling response did not go well: ", err)
				}

				logger.Sugar().Info("responseJson[data]: ", responseJson.Data)

				dataJson := responseJson.Data.AsMap()
				embeddings, ok := dataJson["embeddings"].(string)

				if !ok {
					logger.Sugar().Error("No embeddings found in the return data.")
					embeddings = ""
					// TODO: Handle disagreements?
				}

				// TODO: Make sure to order the embeddings
				responses = append(responses, embeddings)
			}

			wg.Done()
		}()
	}

	wg.Wait()

	target := strings.ToLower(serverAuth)
	endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", serverUrl, target)

	logger.Sugar().Info("Responses: ", responses)

	// TODO: Send proper request
	dataRequest["type"] = "vflAggregateRequest"
	dataRequest["data"] = map[string]any{
		"embeddings": responses,
	}

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []string{}, err
	}

	logger.Sugar().Infof("Sending request to %s.\nEndpoint: %s\nJSON:%v", target, endpoint, string(dataRequestJson))
	responseData, error := sendData(endpoint, dataRequestJson)
	if error != nil {
		logger.Sugar().Errorf("Error sending data to the server, %v", error)
	}

	logger.Sugar().Info("This is the responseData: ", responseData)

	// TODO: Retrieve gradients for each clients, assuming gradients are sent as strings
	// gradients := map[string]string{}
	responses = []string{}

	// TODO: Send the gradients back to the client to update their models
	// for auth, url := range clients {
	// 	wg.Add(1)
	// 	target := strings.ToLower(auth)
	// 	// Construct the end point
	// 	endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)
	// 	gradient := gradients[target]
	//
	// 	// TODO: Send the respective gradient for each client
	// 	logger.Sugar().Infof("Sending gradient descent request to %s.\nEndpoint: %s\nJSON:%v", target, endpoint, string(dataRequest))
	//
	// 	// Async call send the data
	// 	go func() {
	//    dataRequest["type"] = "vflAggregateRequest"
	//    // JSONify dataRequest
	// 		respData, err := sendData(endpoint, dataRequest)
	// 		if err != nil {
	// 			logger.Sugar().Errorf("Error sending data, %v", err)
	//		} else {
	//			responseJson := map[string]string{}
	//			err = json.Unmarshal([]byte(responseData), &responseJson)
	//
	//			if err != nil {
	//				logger.Sugar().Error("Unmarshalling response data did not go well: ", err)
	//			}
	//
	//			responses = append(responses, responseJson)
	//		}
	// 		// Signal that the data request has been sent to all auth providers
	// 		wg.Done()
	// 	}()
	// }

	return responses, nil
}

// This runs the entire training cycle
func runVFLTraining(dataRequest map[string]any, authorizedProviders map[string]string, jobId string) []byte {
	clients := map[string]string{}
	var serverUrl string
	var serverAuth string
	var responseData []string
	var wg sync.WaitGroup

	for auth, url := range authorizedProviders {
		logger.Sugar().Info("AuthorizedProviders url: ", url, " auth: ", auth, " tolower Auth: ", strings.ToLower(auth))
		if strings.ToLower(auth) == "server" {
			serverUrl = url
			serverAuth = auth
		} else if url != "" {
			clients[auth] = url
		}
	}

	logger.Sugar().Info(clients, " and ", serverUrl, " and ", serverAuth)

	for i := range 1 {
		logger.Sugar().Info("Running VFL training round ", i)

		response, err := runVFLTrainingRound(dataRequest, clients, serverAuth, serverUrl)
		responseData = response

		if err != nil {
			logger.Sugar().Error("Training round returned an error.")
			break
		}
	}

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []byte{}
	}

	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflShutdownRequest/%s", url, target)

		go func() {
			sendData(endpoint, dataRequestJson)
			wg.Done()
		}()
	}

	wg.Wait()

	response := map[string]any{
		"jobId":     jobId,
		"responses": responseData,
	}

	return cleanupAndMarshalResponse(response)
}

// Use the data request that was previously built and send it to the authorised providers
// acquired from the request approval
func sendDataToAuthProviders(dataRequest []byte, authorizedProviders map[string]string, msgType string, jobId string) []byte {
	// Setup the wait group for async data requests
	var wg sync.WaitGroup
	var responses []string

	// This will be replaced with AMQ in the future
	agentPort := "8080"
	// Iterate over each auth provider
	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		// Construct the end point
		endpoint := fmt.Sprintf("http://%s:%s/agent/v1/%s/%s", url, agentPort, msgType, target)

		logger.Sugar().Infof("Sending request to %s.\nEndpoint: %s\nJSON:%v", target, endpoint, string(dataRequest))

		// Async call send the data
		go func() {
			respData, err := sendData(endpoint, dataRequest)
			if err != nil {
				logger.Sugar().Errorf("Error sending data, %v", err)
			}
			responses = append(responses, respData)
			// Signal that the data request has been sent to all auth providers
			wg.Done()
		}()
	}

	// Wait until all the requests are complete
	wg.Wait()
	logger.Sugar().Debug("Returning responses")

	responseMap := map[string]any{
		"jobId":     jobId,
		"responses": responses,
	}

	// jsonResponse, _ := json.Marshal(responseMap)
	// return jsonResponse
	return cleanupAndMarshalResponse(responseMap)
}

// Now assumes input is map[string]interface{} and directly marshals it to prettified JSON.
func cleanupAndMarshalResponse(responseMap map[string]any) []byte {
	prettifiedJSON, err := json.MarshalIndent(responseMap, "", "    ")
	if err != nil {
		logger.Sugar().Errorf("Error marshalling cleaned response: %v", err)
	}
	return prettifiedJSON
}

func sendData(endpoint string, jsonData []byte) (string, error) {
	// FIXME: Change to an actual token in the future?
	headers := map[string]string{
		"Authorization": "bearer 1234",
	}
	body, err := api.PostRequest(endpoint, string(jsonData), headers)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func availableProvidersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger.Debug("Starting requestApprovalHandler")
		var availableProviders = make(map[string]lib.AgentDetails)
		resp, err := getAvailableProviders()
		if err != nil {
			logger.Sugar().Errorf("Error getting available providers: %v", err)
			return
		}

		// Bind resp to availableProviders
		availableProviders = resp

		jsonResponse, err := json.Marshal(availableProviders)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling result, %v", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
		w.Write(jsonResponse)
	}
}

// Maybe this should be moved into the orchestrarot
func getAvailableProviders() (map[string]lib.AgentDetails, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Get the value from etcd.
	resp, err := etcdClient.Get(ctx, "/agents/online", clientv3.WithPrefix())
	if err != nil {
		logger.Sugar().Errorf("failed to get value from etcd: %v", err)
		return nil, err
	}

	// Initialize an empty map to store the unmarshaled structs.
	result := make(map[string]lib.AgentDetails)
	// Iterate through the key-value pairs and unmarshal the values into structs.
	for _, kv := range resp.Kvs {
		var target lib.AgentDetails
		err = json.Unmarshal(kv.Value, &target)
		if err != nil {
			// return nil, fmt.Errorf("failed to unmarshal JSON for key %s: %v", key, err)
		}
		result[string(target.Name)] = target
	}

	return result, nil

}
