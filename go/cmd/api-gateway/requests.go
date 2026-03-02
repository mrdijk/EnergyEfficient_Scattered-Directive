// This file contains the handlers for the requests that the API Gateway receives from the client
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Jorrit05/DYNAMOS/pkg/api"
	"github.com/Jorrit05/DYNAMOS/pkg/lib"
	pb "github.com/Jorrit05/DYNAMOS/pkg/proto"
	"github.com/google/uuid"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opencensus.io/trace"
)

const (
	StatusPending = "pending"
	StatusDone    = "done"
	StatusFailed  = "failed"
)

var (
	activeJobID      string
	activeJobLock    sync.Mutex   // to allow only 1 active job at any time
	trainingRequests = sync.Map{} // map[string]TrainingRequestData
)

type TrainingData struct {
	Status string
	Data   *ExperimentData
	// Metadata map[string]any
	// add more fields as needed
}

type TrainingRequestData struct {
	LearningRate float64   `json:"learning_rate"`
	Epochs       int       `json:"epochs"`
	BatchSize    int       `json:"batch_size"`
	Partition    Partition `json:"partition"`
}

func getTrainingStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger.Sugar().Info("Starting getTrainingStatusHandler")
		requestID := r.URL.Query().Get("id")
		v, ok := trainingRequests.Load(requestID)
		if !ok {
			http.Error(w, "Request ID not found", http.StatusNotFound)
			return
		}

		logger.Sugar().Debug("Found training request: ", requestID)
		reqData := v.(TrainingData)
		resp := map[string]any{
			"request_id": requestID,
			"status":     reqData.Status,
			"data":       reqData.Data,
		}
		respBytes, _ := json.MarshalIndent(resp, "", "    ")
		w.WriteHeader(http.StatusOK)
		w.Write(respBytes)
	}
}

type ClientMetrics struct {
	ClientID     string
	Accuracy     float64
	TrainingTime float64 // in milliseconds
}

type RoundMetrics struct {
	GlobalAccuracy    float64
	AggregationTime   float64
	TotalTrainingTime float64
	ClientMetrics     []ClientMetrics
	RoundDuration     time.Duration
}

type ExperimentData struct {
	StartTime string         `json:"start_time"`
	EndTime   string         `json:"end_time"`
	Rounds    []RoundMetrics `json:"rounds"`
	mu        sync.Mutex     `json:"-"`
}

func selectRandomPartitions(partitions []Partition, numClients int) []Partition {
	if len(partitions) <= numClients {
		return partitions
	}

	selected := make([]Partition, len(partitions))
	copy(selected, partitions)

	rand.Shuffle(len(selected), func(i, j int) {
		selected[i], selected[j] = selected[j], selected[i]
	})

	return selected[:numClients]
}

func (er *ExperimentData) SaveToCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"Round", "ClientID", "ClientAccuracy", "ClientTrainingTime_ms",
		"GlobalAccuracy", "AggregationTime_ms", "TotalTrainingTime_ms", "RoundDuration_ms"})

	// Write data
	for roundNum, round := range er.Rounds {
		for _, client := range round.ClientMetrics {
			writer.Write([]string{
				strconv.Itoa(roundNum + 1),
				client.ClientID,
				strconv.FormatFloat(client.Accuracy, 'f', 6, 64),
				strconv.FormatFloat(client.TrainingTime, 'f', 2, 64),
				strconv.FormatFloat(round.GlobalAccuracy, 'f', 6, 64),
				strconv.FormatFloat(round.AggregationTime, 'f', 2, 64),
				strconv.FormatFloat(round.TotalTrainingTime, 'f', 2, 64),
				strconv.FormatInt(round.RoundDuration.Milliseconds(), 10),
			})
		}
	}

	return nil
}

func requestHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		activeJobLock.Lock()
		defer activeJobLock.Unlock()

		// Check for existing active job
		if activeJobID != "" {
			v, _ := trainingRequests.Load(activeJobID)
			reqData := v.(TrainingData)
			resp := map[string]any{
				"error":             "A training job is already in progress.",
				"active_request_id": activeJobID,
				"active_status":     reqData.Status,
			}
			respBytes, _ := json.Marshal(resp)
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write(respBytes)
			return
		}

		// Accept new job
		requestID := uuid.New().String()
		activeJobID = requestID
		reqData := TrainingData{
			Status: StatusPending,
			Data: &ExperimentData{
				// JobID:  requestID,
				Rounds:    make([]RoundMetrics, 0),
				StartTime: time.Now().Format("02-01-06-15:04:05"),
			},
		}
		trainingRequests.Store(requestID, reqData)

		resp := map[string]any{
			"request_id": requestID,
			"status":     StatusPending,
			"start_time": reqData.Data.StartTime,
		}

		logger.Sugar().Info("Accepted new job with id: ", activeJobID)

		// Parse the request body
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

		respBytes, _ := json.Marshal(resp)
		w.WriteHeader(http.StatusAccepted)
		w.Write(respBytes)

		// ---- TRIGGER TRAINING IN BACKGROUND ----
		go func() {
			startTraining(protoRequest, dataRequestInterface, apiReqApproval, r, requestID)
		}()

	}
}

func startTraining(protoRequest *pb.RequestApproval, dataRequestInterface map[string]any, apiReqApproval api.RequestApproval, r *http.Request, requestID string) {
	logger.Debug("Starting training process...")
	// Requests may take up to 10 minutes now
	ctxWithTimeout, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Start a new span with the context that has a timeout
	ctx, span := trace.StartSpan(ctxWithTimeout, "requestApprovalHandler")
	defer span.End()

	// Create a channel to receive the response
	responseChan := make(chan validation)

	requestApprovalMutex.Lock()
	requestApprovalMap[protoRequest.User.Id] = responseChan
	requestApprovalMutex.Unlock()

	_, err := c.SendRequestApproval(ctx, protoRequest)
	if err != nil {
		logger.Sugar().Errorf("error in sending requestapproval: %v", err)
	}

	select {
	case validationStruct := <-responseChan:
		msg := validationStruct.response

		logger.Sugar().Infof("Received response, %s", msg.Type)
		if msg.Type != "requestApprovalResponse" {
			logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
			// http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		requestMetadata := &pb.RequestMetadata{
			JobId: msg.JobId,
		}
		dataRequestInterface["requestMetadata"] = requestMetadata

		logger.Sugar().Infof("Data Prepared jsonData: %s", dataRequestInterface)

		var response []byte

		if apiReqApproval.Type == "hflTrainModelRequest" {
			ctxWithoutCancel := context.WithoutCancel(r.Context())
			response = runHFLTraining(dataRequestInterface, msg.AuthorizedProviders, msg.JobId, ctxWithoutCancel, requestID)

		} else {
			// Marshal the combined data back into JSON for forwarding
			dataRequestJson, err := json.Marshal(dataRequestInterface)
			if err != nil {
				logger.Sugar().Errorf("Error marshalling combined data: %v", err)
				return
			}

			response = sendDataToAuthProviders(dataRequestJson, msg.AuthorizedProviders, apiReqApproval.Type, msg.JobId)
		}

		// w.WriteHeader(http.StatusOK)
		// w.Write(response)
		logger.Sugar().Info("Training process completed for request id: ", requestID)
		logger.Sugar().Info("Response: ", string(response))
		return

	case <-ctx.Done():
		// http.Error(w, "Request timed out", http.StatusRequestTimeout)
		return
	}

}

func runHFLTrainingRound(dataRequest map[string]any, clients map[string]string, serverAuth, serverUrl string, learningRate float64) (RoundMetrics, error) {
	var wg sync.WaitGroup
	var mu sync.Mutex // Add mutex
	clientUpdates := map[string]string{}
	clientMetricsList := []ClientMetrics{}
	var total_time_per_round float64 = 0.0

	start := time.Now()

	// Ask each client to train locally
	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)

		ips, err := net.LookupIP(url)
		if err == nil && len(ips) != 0 {
			url = ips[0].String()
		}

		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", url, target)
		dataRequest["type"] = "hflTrainRequest"
		dataRequest["data"] = map[string]any{
			"learning_rate": learningRate,
		}

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return RoundMetrics{}, err
		}

		go func(auth string, endpoint string) {
			defer wg.Done()
			logger.Sugar().Infof("Sending training request to: %s", auth)
			responseData, err := sendData(endpoint, dataRequestJson)
			if err != nil {
				logger.Sugar().Errorf("Error sending data to client %s: %v", auth, err)
				return
			}
			responseJson := &pb.MicroserviceCommunication{}
			err = json.Unmarshal([]byte(responseData), responseJson)
			if err != nil {
				logger.Sugar().Errorf("Error unmarshalling client %s response: %v", auth, err)
				return
			}
			dataJson := responseJson.Data.AsMap()
			modelUpdate, ok := dataJson["model_update"].(string)

			if !ok {
				logger.Sugar().Errorf("No model_update found in client %s response.", auth)
				return // Return early if no model update
			}

			client_accuracy := dataJson["accuracy"].(float64)
			training_duration := dataJson["t_train"].(float64)

			// Use mutex to update everything at once
			mu.Lock()
			total_time_per_round += training_duration
			clientMetricsList = append(clientMetricsList, ClientMetrics{
				ClientID:     auth,
				Accuracy:     client_accuracy,
				TrainingTime: training_duration,
			})
			clientUpdates[strings.ToLower(auth)] = modelUpdate // Only update once, inside mutex
			mu.Unlock()

			logger.Sugar().Infof("Received update from %s: %d bytes", auth, len(modelUpdate))
		}(auth, endpoint)
	}

	wg.Wait()

	// logger.Sugar().Infof("Collected %d client updates from %d clients", len(clientUpdates), len(clients))
	// for auth, update := range clientUpdates {
	// 	logger.Sugar().Debugf("Client %s: update size = %d bytes", auth, len(update))
	// }

	// logger.Sugar().Infof("Total training time for this round[ms]: %f", total_time_per_round)

	// Send all client model updates to server for aggregation
	target := strings.ToLower(serverAuth)
	serverEndpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", serverUrl, target)

	updateList := []string{}
	for _, update := range clientUpdates {
		updateList = append(updateList, update)
	}

	logger.Sugar().Infof("Sending %d updates to server for aggregation", len(updateList))

	dataRequest["type"] = "hflAggregateRequest"
	dataRequest["data"] = map[string]any{
		"model_updates": updateList,
	}

	// Debug: log the structure
	logger.Sugar().Debugf("Data request structure: type=%s, updates_count=%d",
		dataRequest["type"], len(updateList))

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling server aggregation request: %v", err)
		return RoundMetrics{}, err
	}

	// logger.Sugar().Debugf("Request JSON size: %d bytes", len(dataRequestJson))

	responseData, err := sendData(serverEndpoint, dataRequestJson)
	if err != nil {
		logger.Sugar().Errorf("Error sending aggregation request to server: %v", err)
		return RoundMetrics{}, err
	}

	serverResponse := &pb.MicroserviceCommunication{}
	err = json.Unmarshal([]byte(responseData), serverResponse)
	if err != nil {
		logger.Sugar().Error("Unmarshalling server response failed: ", err)
	}

	// Debug what we received
	// logger.Sugar().Infof("Server response data fields: %+v", serverResponse.Data.GetFields())

	accuracy := serverResponse.Data.GetFields()["accuracy"].GetNumberValue()
	aggregation_duration := serverResponse.Data.GetFields()["t_agg"].GetNumberValue()
	globalParams := serverResponse.Data.GetFields()["global_params"].GetStringValue()

	// logger.Sugar().Infof("GlobalParams length: %d bytes", len(globalParams))
	// logger.Sugar().Infof("GlobalParams first 100 chars: %s", globalParams[:min(100, len(globalParams))])

	if globalParams == "" {
		logger.Sugar().Error("- Received empty params from server")
	}

	// Send the new global model to all clients
	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", url, target)

		dataRequest["type"] = "hflLoadGlobalModel"
		dataRequest["data"] = map[string]any{
			"global_params": globalParams,
		}

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling global model broadcast: %v", err)
			return RoundMetrics{}, err
		}

		go func(auth string, endpoint string) {
			defer wg.Done()
			logger.Sugar().Infof("Sending global model to %s", auth)
			response, err := sendData(endpoint, dataRequestJson)
			if err != nil {
				logger.Sugar().Errorf("Error sending global model to client %s: %v, response: %s", auth, err, response)
			}
		}(auth, endpoint)
	}

	wg.Wait()
	// duration := time.Duration(time.Since(start)).Milliseconds()
	duration := time.Since(start)
	logger.Sugar().Infof("Round duration in ms: %f", duration)

	return RoundMetrics{
		GlobalAccuracy:    accuracy,
		AggregationTime:   aggregation_duration,
		TotalTrainingTime: total_time_per_round,
		ClientMetrics:     clientMetricsList,
		RoundDuration:     duration,
	}, nil

}

func runHFLTraining(dataRequest map[string]any, authorizedProviders map[string]string, jobId string, ctx context.Context, requestID string) []byte {
	clients := map[string]string{}
	var serverUrl string
	var serverAuth string
	var finalAccuracy float64
	var cycles int = 10
	var learningRate float64 = 0.01
	// var change_policies int64 = -1
	var dataProviders []string = []string{}

	// Zipf Partition Configuration
	var nrPartitions int = 20
	var totalRows int = 531130
	// iid represents the number of classes present in the client datasets
	var sigma_iid int = 0
	// ed represents the distribution of dataset sizes
	// Large sigma represent a balanced distribution as sigma aproaches 1 the distribution becomes more skewed
	var sigma_ed float64 = 0
	var seed int64 = time.Now().UnixNano()

	var wg sync.WaitGroup
	results := &ExperimentData{
		// JobID:  jobId,
		Rounds: make([]RoundMetrics, 0),
	}
	data, ok := dataRequest["data"].(map[string]any)
	logger.Sugar().Info("Data from req: ", data)

	if ok {
		if val, ok := data["cycles"].(float64); ok {
			cycles = int(val)
		}
		// if val, ok := data["learning_rate"].(float64); ok {
		// 	learningRate = val
		// }
		if val, ok := data["partitions"].(int); ok {
			nrPartitions = val
		}
		if val, ok := data["iid"].(int); ok {
			sigma_iid = val
		}
		if val, ok := data["ed"].(float64); ok {
			sigma_ed = val
		}
	}

	partitionConfig := makePartitionConfiguration(nrPartitions, totalRows, sigma_ed, seed)
	trainingFailed := false

	// Check first partition
	// fmt.Printf("Partition 1: RowCount=%d, len(RowIDs)=%d\n",
	// 	partitionConfig.Partitions[0].RowCount,
	// 	len(partitionConfig.Partitions[0].RowIDs))
	for auth, url := range authorizedProviders {
		if strings.ToLower(auth) == "server" {
			serverUrl = url
			serverAuth = auth
		} else if url != "" {
			clients[auth] = url
		}
		dataProviders = append(dataProviders, auth)
	}

	// Select partitions for clients
	numClients := len(clients)
	selectedPartitions := selectRandomPartitions(partitionConfig.Partitions, numClients)

	// logger.Sugar().Infof("Selected %d partitions for %d clients", len(selectedPartitions), numClients)
	// logger.Sugar().Info("Sending ping to start pods...")

	// The server has a test set with 26033 samples
	serverRowIDs := make([]int, 26033)
	for i := range serverRowIDs {
		serverRowIDs[i] = i
	}

	// Partition for the server contains 100% the data from the test set
	// Since there is only one partition for this dataset it gets rank 1
	serverPartition := Partition{
		Rank:        1,
		RowCount:    26032,
		Probability: 1,
		Percentage:  100,
		RowIDs:      serverRowIDs,
	}

	// Map clients to partitions
	clientPartitionMap := make(map[string]Partition)
	clientIdx := 0
	for auth := range clients {
		if clientIdx < len(selectedPartitions) {
			clientPartitionMap[auth] = selectedPartitions[clientIdx]
			clientIdx++
		}
	}

	user, ok := dataRequest["user"].(*pb.User)
	if !ok {
		logger.Sugar().Error("Did not retrieve User from dataRequest, cannot dynamically verify each training round.")
		user = &pb.User{}
	}

	var noPing bool = false

	// Send pring to each client with their assigned partition
	for auth, url := range authorizedProviders {
		wg.Add(1)

		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", url, target)

		// Get this clients partition
		partition, hasPartition := clientPartitionMap[auth]

		if target == "server" {
			partition, hasPartition = serverPartition, true
		}

		go func(auth string, endpoint string, partition Partition, hasPartition bool) {
			defer wg.Done()
			if !hasPartition {
				logger.Sugar().Warnf("No partition assigned to client %s", auth)
				// return
			}
			logger.Sugar().Infof("Sending Ping to %s with partition %d (%d rows)", auth, partition.Rank, partition.RowCount)

			// Create request with single partition for this client
			dataRequest["type"] = "hflPingRequest"
			dataRequest["data"] = map[string]any{
				"partition": partition, // Single partition, not array
				"iid":       sigma_iid,
			}

			dataRequestJson, err := json.Marshal(dataRequest)
			if err != nil {
				logger.Sugar().Errorf("Error marshalling ping request for %s: %v", auth, err)
				return
			}

			for i := range 5 {
				_, err := sendData(endpoint, dataRequestJson)

				// If Ping resquest arrives correctly we don't need to try again
				if err == nil {
					logger.Sugar().Infof("Successfully pinged %s with partition %d", auth, partition.Rank)
					break
				}
				// After 5 tries give up and continue
				if i == 4 {
					noPing = true
					logger.Sugar().Errorf("Failed to ping %s after 5 attempts", auth)
					break
				}
			}
		}(auth, endpoint, partition, hasPartition)
	}

	wg.Wait()

	// If a client doesn't start, stop training and set status to "Failed" and shutdown all clients
	if noPing {
		logger.Sugar().Error("No ping from one or more clients")
		v, ok := trainingRequests.Load(requestID)

		if !ok {
			logger.Sugar().Error("Could not find the training request to update status.")
			return []byte{}
		}

		reqData := v.(TrainingData)
		reqData.Status = StatusFailed
		trainingRequests.Store(requestID, reqData)

		logger.Sugar().Infow("Training failed", "requestID", requestID, "status", reqData.Status)
		failed_response := map[string]any{
			"request_id": requestID,
			"status":     reqData.Status,
		}

		// Release the active job lock
		activeJobLock.Lock()
		activeJobID = ""
		activeJobLock.Unlock()

		dataRequest["type"] = "hflShutdownRequest"
		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling shutdown request: %v", err)
			return []byte{}
		}

		for auth, url := range authorizedProviders {
			wg.Add(1)
			target := strings.ToLower(auth)
			endpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", url, target)

			go func(auth, endpoint string) {
				logger.Sugar().Infof("-- Sending shutdown request to -> %s ", auth)
				sendData(endpoint, dataRequestJson)
				wg.Done()
			}(auth, endpoint)
		}

		wg.Wait()

		// return []byte{}
		return cleanupAndMarshalResponse(failed_response)
	}

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("-=-=-=-=-=-=Starting training-=-=-=-=-=-=-=-=")
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Running HFL for ", cycles, " rounds")
	start := time.Now().Format(time.RFC3339)
	results.StartTime = start

TrainLoop:
	for round := int(1); round <= cycles; round++ {
		logger.Sugar().Info("Running HFL training round ", round)

		protoRequest := &pb.RequestApproval{
			Type:             "hflTrainModelRequest",
			User:             user,
			DataProviders:    dataProviders,
			DestinationQueue: "policyEnforcer-in",
		}

		// Create a channel to receive the response
		responseChan := make(chan validation)

		requestApprovalMutex.Lock()
		requestApprovalMap[protoRequest.User.Id] = responseChan
		requestApprovalMutex.Unlock()

		noValidation := false

		logger.Sugar().Info("- Sending policy reverification request")
		for i := range 5 {
			_, err := c.SendRequestApproval(ctx, protoRequest)
			if err == nil {
				break
			}

			logger.Sugar().Warnf("error in sending/receiving requestApproval: %v", err)

			if i == 4 {
				noValidation = true
			}
		}

		if noValidation {
			logger.Sugar().Error("No reverification approval received, error in network. Shutting down operation.")
			break TrainLoop
		}

		validationStruct := <-responseChan
		msg := validationStruct.response
		logger.Sugar().Info("Received validation message: ", msg, ", with vstruct: ", validationStruct)

		if msg.Type != "requestApprovalResponse" {
			logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
			return []byte{}
		}

		if msg.Error != "" || len(msg.AuthorizedProviders) != len(authorizedProviders) {
			logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
			logger.Sugar().Info("   Policy does not allow this training to continue.")
			logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
			break TrainLoop
		}

		logger.Sugar().Info("Sending training requests")
		RoundMetrics, err := runHFLTrainingRound(dataRequest, clients, serverAuth, serverUrl, learningRate)
		if err != nil {
			logger.Sugar().Errorf("Round %d failed: %v", round, err)
			trainingFailed = true
			break TrainLoop
		}
		results.Rounds = append(results.Rounds, RoundMetrics)

		logger.Sugar().Infof("Round %d complete - Global Accuracy: %.4f, Duration[s]: %.4f", round, RoundMetrics.GlobalAccuracy, RoundMetrics.RoundDuration.Seconds())
		finalAccuracy = RoundMetrics.GlobalAccuracy

		if trainingFailed {
			break
		}
	}

	end := time.Now().Format(time.RFC3339)
	results.EndTime = end

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Final accuracy achieved: ", finalAccuracy)
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

	response := map[string]any{
		"jobId":    jobId,
		"accuracy": finalAccuracy,
		"start":    start,
		"end":      end,
	}

	// --- SET STATUS and UNLOCK ---
	v, ok := trainingRequests.Load(requestID)

	if !ok {
		logger.Sugar().Error("Could not find the training request to update status.")
		return []byte{}
	}

	reqData := v.(TrainingData)
	reqData.Data = results
	if trainingFailed {
		reqData.Status = StatusFailed
	} else {
		reqData.Status = StatusDone
	}
	trainingRequests.Store(requestID, reqData)
	logger.Sugar().Infow("Job completed", "requestID", requestID, "status", reqData.Status, "accuracy", finalAccuracy)
	new_response := map[string]any{
		"request_id": requestID,
		"status":     reqData.Status,
		"data":       reqData.Data,
	}

	// Release the active job lock
	activeJobLock.Lock()
	activeJobID = ""
	activeJobLock.Unlock()

	// Marshal and return
	responseJson, err := json.MarshalIndent(new_response, "", "    ")
	if err != nil {
		logger.Sugar().Errorf("Error marshalling training results: %v", err)
		return []byte{}
	}

	logger.Sugar().Infof("Training results: ", string(responseJson))

	dataRequest["type"] = "hflShutdownRequest"
	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling shutdown request: %v", err)
		return []byte{}
	}

	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/hflTrainRequest/%s", url, target)

		go func(auth, endpoint string) {
			logger.Sugar().Infof("-- Sending shutdown request to -> %s ", auth)
			sendData(endpoint, dataRequestJson)
			wg.Done()
		}(auth, endpoint)
	}

	wg.Wait()
	return cleanupAndMarshalResponse(response) // note this is not the same as responseJson
}

func runVFLTrainingRound(dataRequest map[string]any, clients map[string]string, serverAuth string, serverUrl string, learning_rate float64) (float64, error) {
	var wg sync.WaitGroup
	responses := map[string]string{}

	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)

		ips, err := net.LookupIP(url)
		if err == nil && len(ips) != 0 {
			url = ips[0].String()
		}

		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		dataRequest["type"] = "vflTrainRequest"

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return 0., err
		}

		go func() {
			responseData, err := sendData(endpoint, dataRequestJson)

			if err != nil {
				logger.Sugar().Errorf("Error sending data, %v", err)
			} else {
				responseJson := &pb.MicroserviceCommunication{}
				err = json.Unmarshal([]byte(responseData), responseJson)

				if err != nil {
					logger.Sugar().Error("Unmarshalling response did not go well: ", err)
				}

				dataJson := responseJson.Data.AsMap()
				embeddings, ok := dataJson["embeddings"].(string)

				if !ok {
					logger.Sugar().Error("No embeddings found in the return data.")
					embeddings = ""
					// TODO: Handle disagreements?
				}

				responses[target] = embeddings
			}

			wg.Done()
		}()
	}

	wg.Wait()

	target := strings.ToLower(serverAuth)
	endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", serverUrl, target)

	dataRequest["type"] = "vflAggregateRequest"
	dataRequest["data"] = map[string]any{
		"embeddings": []string{responses["clientone"], responses["clienttwo"], responses["clientthree"]},
	}

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return 0., err
	}

	responseData, error := sendData(endpoint, dataRequestJson)
	if error != nil {
		logger.Sugar().Errorf("Error sending data to the server, %v", error)
	}

	serverResponse := &pb.MicroserviceCommunication{}
	err = json.Unmarshal([]byte(responseData), serverResponse)

	if err != nil {
		logger.Sugar().Error("Unmarshalling response did not go well: ", err)
	}

	accuracy := serverResponse.Data.GetFields()["accuracy"].GetNumberValue()
	gradientList := serverResponse.Data.GetFields()["gradients"].GetListValue().GetValues()

	gradients := []string{}
	for _, val := range gradientList {
		gradients = append(gradients, val.GetStringValue())
	}

	// TODO: Send the gradients back to the client to update their models
	index := 0
	for auth, url := range clients {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		dataRequest["type"] = "vflGradientDescentRequest"
		dataRequest["data"] = map[string]any{
			"gradients":     gradients[index],
			"learning_rate": learning_rate,
		}

		index++

		dataRequestJson, err := json.Marshal(dataRequest)
		if err != nil {
			logger.Sugar().Errorf("Error marshalling combined data: %v", err)
			return 0., err
		}

		go func() {
			response, err := sendData(endpoint, dataRequestJson)
			if err != nil {
				logger.Sugar().Error("Error sending data, ", err, ", received: ", response)
			}
			wg.Done()
		}()
	}

	wg.Wait()

	return accuracy, nil
}

func runVFLTraining(dataRequest map[string]any, authorizedProviders map[string]string, jobId string, ctx context.Context) []byte {
	clients := map[string]string{}
	var serverUrl string
	var serverAuth string
	var finalAccuracy float64
	var wg sync.WaitGroup

	var cycles int64 = 10
	var learning_rate float64 = 0.05
	var change_policies int64 = -1
	var dataProviders []string = []string{}

	data, ok := dataRequest["data"].(map[string]any)
	logger.Sugar().Info("Data from req: ", data)

	if ok {
		floatCycles, ok := data["cycles"].(float64)

		if ok {
			cycles = int64(floatCycles)
		}

		floatLearningRate, ok := data["learning_rate"].(float64)
		if ok {
			learning_rate = floatLearningRate
		}

		changePolicies, ok := data["change_policies"].(float64)
		if ok {
			change_policies = int64(changePolicies)
		}
	}

	for auth, url := range authorizedProviders {
		if strings.ToLower(auth) == "server" {
			serverUrl = url
			serverAuth = auth
		} else if url != "" {
			clients[auth] = url
		}

		dataProviders = append(dataProviders, auth)
	}

	logger.Sugar().Info("Sending ping to start pods...")
	dataRequest["type"] = "vflPingRequest"

	dataRequestJson, err := json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []byte{}
	}

	user, ok := dataRequest["user"].(*pb.User)

	if !ok {
		logger.Sugar().Info("Did not retrieve User from dataRequest, cannot dynamically verify each training round.")
		user = &pb.User{}
	}

	var noPing bool = false

	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		go func() {
			// TODO: Repeat ping until no error, after 5 tries, cancel request
			for i := range 5 {
				_, err := sendData(endpoint, dataRequestJson)

				if err == nil {
					break
				}

				if i == 4 {
					noPing = true
				}
			}

			wg.Done()
		}()
	}

	if noPing {
		logger.Sugar().Error("No ping from a client or the server. Something is wrong.")
	}

	wg.Wait()

	logger.Sugar().Info("Running VFL for ", cycles, " rounds")
	for round := range cycles {
		logger.Sugar().Info("Running VFL training round ", round)

		// TODO: Implement policy change request
		if change_policies == round {
			logger.Sugar().Info("Sending in the policy change request, removing client 3 from the agreement.")
			logger.Sugar().Info("TODO: Policy change request not yet implemented.")

			policyUpdate := &pb.RequestApproval{
				Type:             "policyRemoval",
				User:             user,
				DestinationQueue: "policyEnforcer-in",
			}

			// Create a channel to receive the response
			responseChan := make(chan validation)

			requestApprovalMutex.Lock()
			requestApprovalMap[policyUpdate.User.Id] = responseChan
			requestApprovalMutex.Unlock()

			logger.Sugar().Info("- Sending policy removal request")
			_, err = c.SendRequestApproval(ctx, policyUpdate)
			if err != nil {
				logger.Sugar().Warnf("error in sending/receiving policy removal: %v", err)
			}
		}

		protoRequest := &pb.RequestApproval{
			Type:             "vflTrainModelRequest",
			User:             user,
			DataProviders:    dataProviders,
			DestinationQueue: "policyEnforcer-in",
		}

		// Create a channel to receive the response
		responseChan := make(chan validation)

		requestApprovalMutex.Lock()
		requestApprovalMap[protoRequest.User.Id] = responseChan
		requestApprovalMutex.Unlock()

		noValidation := false

		logger.Sugar().Info("- Sending policy reverification request")
		for i := range 5 {
			_, err = c.SendRequestApproval(ctx, protoRequest)
			if err != nil {
				logger.Sugar().Warnf("error in sending/receiving requestApproval: %v", err)
			}

			if err == nil {
				break
			}

			if i == 4 {
				noValidation = true
			}
		}

		if noValidation {
			logger.Sugar().Error("No reverification approval received, error in network. Shutting down operation.")
			break
		}

		select {
		case validationStruct := <-responseChan:
			msg := validationStruct.response
			logger.Sugar().Info("Received validation message: ", msg, ", with vstruct: ", validationStruct)

			if msg.Type != "requestApprovalResponse" {
				logger.Sugar().Errorf("Unexpected message received, type: %s", msg.Type)
				return []byte{}
			}

			if msg.Error != "" || len(msg.AuthorizedProviders) != len(authorizedProviders) {
				logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
				logger.Sugar().Info("   Policy does not allow this training to continue.")
				logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
				break
			}

			logger.Sugar().Info("- Sending training request")
			accuracy, err := runVFLTrainingRound(dataRequest, clients, serverAuth, serverUrl, learning_rate)
			logger.Sugar().Info("- Intermediate accuracy achieved: ", accuracy, " for round ", round)
			finalAccuracy = accuracy

			if err != nil {
				logger.Sugar().Error("Training round returned an error.")
				break
			}
			// default:
		}
	}

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Final accuracy achieved: ", finalAccuracy)
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

	dataRequest["type"] = "vflShutdownRequest"

	dataRequestJson, err = json.Marshal(dataRequest)
	if err != nil {
		logger.Sugar().Errorf("Error marshalling combined data: %v", err)
		return []byte{}
	}

	for auth, url := range authorizedProviders {
		wg.Add(1)
		target := strings.ToLower(auth)
		endpoint := fmt.Sprintf("http://%s:8080/agent/v1/vflTrainRequest/%s", url, target)

		go func() {
			sendData(endpoint, dataRequestJson)
			wg.Done()
		}()
	}

	wg.Wait()

	response := map[string]any{
		"jobId":    jobId,
		"accuracy": finalAccuracy,
	}

	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
	logger.Sugar().Info("Final accuracy achieved: ", finalAccuracy)
	logger.Sugar().Info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

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
	logger.Sugar().Infof("POST -> %s", endpoint)
	body, err := api.PostRequest(endpoint, string(jsonData), headers)
	if err != nil {
		logger.Sugar().Errorf("POST error to %s: %v", endpoint, err)
		return "", err
	} else {
		logger.Sugar().Infof("POST OK from %s: %s", endpoint, body)
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
