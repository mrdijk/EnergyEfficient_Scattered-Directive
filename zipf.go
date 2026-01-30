package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// PartitionConfiguration represents a configuration with N partitions
type PartitionConfiguration struct {
	NumPartitions int
	TotalRows     int
	Exponent      float64
	Partitions    []Partition
}

// Partition represents a single partition with its row assignment
type Partition struct {
	ID          int     // Partition ID (1 to N)
	Rank        int     // Same as ID for Zipf
	RowCount    int     // Number of rows assigned
	Probability float64 // Zipf probability
	Percentage  float64 // Percentage of total rows
	RowIDs      []int   // Actual row IDs assigned to this partition
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
func (zpb *ZipfPartitionBuilder) BuildConfiguration(numPartitions, totalRows int, s float64) *PartitionConfiguration {
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
			ID:          rank,
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

// PrintConfigurationSummary prints a summary of the configuration
func (zpb *ZipfPartitionBuilder) PrintConfigurationSummary(config *PartitionConfiguration) {
	fmt.Printf("\n%-10s %-12s %-12s %-10s\n", "Partition", "Rows", "Probability", "Percentage")
	fmt.Println(strings.Repeat("-", 50))

	displayCount := min(config.NumPartitions, 10)

	for i := range displayCount {
		p := config.Partitions[i]
		fmt.Printf("%-10d %-12d %-12.6f %-10.2f%%\n",
			p.ID, p.RowCount, p.Probability, p.Percentage)
	}

	if config.NumPartitions > displayCount {
		fmt.Printf("... (%d more partitions)\n", config.NumPartitions-displayCount)

		// Show last 3
		fmt.Println()
		for i := config.NumPartitions - 3; i < config.NumPartitions; i++ {
			p := config.Partitions[i]
			fmt.Printf("%-10d %-12d %-12.6f %-10.2f%%\n",
				p.ID, p.RowCount, p.Probability, p.Percentage)
		}
	}

	// Calculate concentration metrics
	top3 := 0
	top10 := 0
	for i := 0; i < config.NumPartitions; i++ {
		if i < 3 {
			top3 += config.Partitions[i].RowCount
		}
		if i < 10 {
			top10 += config.Partitions[i].RowCount
		}
	}

	fmt.Printf("\n📊 Data Concentration:\n")
	fmt.Printf("   Top 3 partitions:  %.2f%% of data\n", float64(top3)/float64(config.TotalRows)*100)
	if config.NumPartitions >= 10 {
		fmt.Printf("   Top 10 partitions: %.2f%% of data\n", float64(top10)/float64(config.TotalRows)*100)
	}
}

// PrintComparisonTable prints a comparison across all configurations
func PrintComparisonTable(configs []*PartitionConfiguration) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("COMPARISON ACROSS ALL CONFIGURATIONS")
	fmt.Println(strings.Repeat("=", 100))

	fmt.Printf("\n%-15s %-12s %-15s %-15s %-15s\n",
		"Num Partitions", "Total Rows", "Top-1 %", "Top-3 %", "Top-10 %")
	fmt.Println(strings.Repeat("-", 80))

	for _, config := range configs {
		top1 := float64(config.Partitions[0].RowCount) / float64(config.TotalRows) * 100

		top3 := 0
		for i := 0; i < 3 && i < len(config.Partitions); i++ {
			top3 += config.Partitions[i].RowCount
		}
		top3Pct := float64(top3) / float64(config.TotalRows) * 100

		top10 := 0
		for i := 0; i < 10 && i < len(config.Partitions); i++ {
			top10 += config.Partitions[i].RowCount
		}
		top10Pct := float64(top10) / float64(config.TotalRows) * 100

		fmt.Printf("%-15d %-12d %-15.2f%% %-15.2f%% %-15.2f%%\n",
			config.NumPartitions, config.TotalRows, top1, top3Pct, top10Pct)
	}
}

func main() {
	// Your specified partition counts
	partitionCounts := []int{15, 30, 60, 90, 120, 150, 190, 230, 260, 300, 330, 360, 400}

	// Configuration
	var totalRows int = 100000
	var zipfExponent float64 = 2.0
	var seed int64 = time.Now().UnixNano()

	fmt.Println("=== Zipf-based Dataset Partitioning with Varying Partition Counts ===")
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Total Rows per Dataset: %d\n", totalRows)
	fmt.Printf("  Zipf Exponent (s): %.2f\n", zipfExponent)
	fmt.Printf("  Partition Counts: %v\n", partitionCounts)
	fmt.Printf("  Random Seed: %d\n", seed)

	// Create builder
	builder := NewZipfPartitionBuilder(seed)

	// Generate all configurations
	configs := make([]*PartitionConfiguration, len(partitionCounts))

	for i, numPartitions := range partitionCounts {
		fmt.Printf("%s", "\n\n"+strings.Repeat("=", 80)+"\n")
		fmt.Printf("CONFIGURATION %d: %d Partitions\n", i+1, numPartitions)
		fmt.Printf("%s", strings.Repeat("=", 80)+"\n")

		config := builder.BuildConfiguration(numPartitions, totalRows, zipfExponent)
		configs[i] = config

		builder.PrintConfigurationSummary(config)
	}

	// Print comparison table
	PrintComparisonTable(configs)

	// Detailed example with row assignments for one configuration
	fmt.Printf("%s", "\n\n"+strings.Repeat("=", 80)+"\n")
	fmt.Println("DETAILED EXAMPLE: Configuration with 30 partitions (with actual row assignments)")
	fmt.Printf("%s", strings.Repeat("=", 80)+"\n")

	exampleConfig := builder.BuildConfiguration(30, 1000, zipfExponent)
	builder.AssignRowsToPartitions(exampleConfig)

	fmt.Printf("\nShowing first 5 partitions with sample row IDs:\n\n")
	for i := 0; i < 5 && i < len(exampleConfig.Partitions); i++ {
		p := exampleConfig.Partitions[i]
		fmt.Printf("Partition %d: %d rows (%.2f%%)\n", p.ID, p.RowCount, p.Percentage)

		// Show first 10 row IDs
		sampleSize := min(len(p.RowIDs), 10)
		fmt.Printf("  Sample row IDs: ")
		for j := range sampleSize {
			fmt.Printf("%d ", p.RowIDs[j])
		}
		if len(p.RowIDs) > sampleSize {
			fmt.Printf("... (%d more)", len(p.RowIDs)-sampleSize)
		}
		fmt.Println()
	}
}
