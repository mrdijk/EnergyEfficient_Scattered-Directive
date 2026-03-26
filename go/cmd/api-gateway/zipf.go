package main

import (
	"math"
	"math/rand"
)

// Configuration with N partitions
type PartitionConfiguration struct {
	NumPartitions int         `json:"nr_partitions"` // Number of partitions in config (N)
	TotalRows     int         `json:"total_rows"`    // Total nr of rows in this config
	Exponent      float64     `json:"zipf_exp"`      // Zipf exponent
	Partitions    []Partition `json:"partitions"`    // list of all partions in this config
}

// Single partition with its row assignment
type Partition struct {
	Rank        int     `json:"zipf_rank"`   // Partition Rank (1 to N)
	RowCount    int     `json:"row_count"`   // Number of rows assigned
	Probability float64 `json:"Probability"` // Zipf probability
	Percentage  float64 `json:"Percentage"`  // Percentage of total rows
	RowIDs      []int   `json:"row_ids"`     // Actual row IDs assigned to this partition
}

// ZipfPartitionBuilder creates partition configurations
type ZipfPartitionBuilder struct {
	rand *rand.Rand
}

// NewZipfPartitionBuilder creates a new builder
func NewZipfPartitionBuilder(seed int64) *ZipfPartitionBuilder {
	return &ZipfPartitionBuilder{
		rand: rand.New(rand.NewSource(seed)),
	}
}

// CalculateHarmonicNumber computes H(n, s) = sum_{i=1}^{n} 1/i^s
func (zpb *ZipfPartitionBuilder) CalculateHarmonicNumber(n int, s float64) float64 {
	sum := 0.0
	for i := 1; i <= n; i++ {
		sum += 1.0 / math.Pow(float64(i), s)
	}
	return sum
}

// BuildConfiguration creates a partition configuration with row assignments
func (zpb *ZipfPartitionBuilder) BuildConfiguration(numPartitions int, totalRows int, s float64) *PartitionConfiguration {
	config := &PartitionConfiguration{
		NumPartitions: numPartitions,
		TotalRows:     totalRows,
		Exponent:      s,
		Partitions:    make([]Partition, numPartitions),
	}

	// Calculate normalization constant
	hnorm := zpb.CalculateHarmonicNumber(numPartitions, s)

	// Calculate probabilities and expected row counts
	for i := range numPartitions {
		rank := i + 1
		prob := (1.0 / math.Pow(float64(rank), s)) / hnorm
		rowCount := int(math.Round(float64(totalRows) * prob))

		config.Partitions[i] = Partition{
			Rank:        rank,
			RowCount:    rowCount,
			Probability: prob,
			Percentage:  prob * 100,
			RowIDs:      make([]int, 0),
		}
	}

	// Adjust first partition to ensure exact total
	totalAssigned := 0
	for i := range numPartitions {
		totalAssigned += config.Partitions[i].RowCount
	}
	if totalAssigned != totalRows {
		config.Partitions[0].RowCount += (totalRows - totalAssigned)
		config.Partitions[0].Percentage = float64(config.Partitions[0].RowCount) / float64(totalRows) * 100
	}

	return config
}

// AssignRowsToPartitions assigns actual row IDs to partitions
func (zpb *ZipfPartitionBuilder) AssignRowsToPartitions(config *PartitionConfiguration) {
	// Build cumulative distribution
	cumProb := make([]float64, config.NumPartitions)
	sum := 0.0
	for i := 0; i < config.NumPartitions; i++ {
		sum += config.Partitions[i].Probability
		cumProb[i] = sum
	}

	// Assign each row to a partition
	for rowID := 1; rowID <= config.TotalRows; rowID++ {
		r := zpb.rand.Float64()

		// Find which partition this row belongs to
		for i, cp := range cumProb {
			if r <= cp {
				config.Partitions[i].RowIDs = append(config.Partitions[i].RowIDs, rowID)
				break
			}
		}
	}

	// Update actual row counts based on assignments
	for i := 0; i < config.NumPartitions; i++ {
		config.Partitions[i].RowCount = len(config.Partitions[i].RowIDs)
		config.Partitions[i].Percentage = float64(config.Partitions[i].RowCount) / float64(config.TotalRows) * 100
	}
}

func makePartitionConfiguration(nrPartitions int, totalRows int, sigma_ed float64, seed int64) *PartitionConfiguration {
	// Create builder
	builder := NewZipfPartitionBuilder(seed)

	// Generate all partitions
	// Convert exponent to same sigma as in Drainakis et al.
	// Correct zipfExponent = 1 / σ
	// σ = 1.7 → s = 0.588
	// σ = 2.0 → s = 0.500
	// σ = 2.3 → s = 0.435
	// σ = 1000 → s = 0.001
	zipfExponent := 1 / sigma_ed
	config := builder.BuildConfiguration(nrPartitions, totalRows, zipfExponent)
	// And assign row IDs
	builder.AssignRowsToPartitions(config)

	return config
}
