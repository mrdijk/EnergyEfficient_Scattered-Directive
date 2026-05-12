# import argparse
# import os

import pandas as pd
from scipy.stats import shapiro

if __name__ == "__main__":
    df = pd.read_csv(
        "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/K05/combined_global_stats.csv",
        index_col=0,
    )

    columns_to_test = ["AggregationTime", "TotalTrainingTime", "RoundDuration"]
    non_normal = {col: 0 for col in columns_to_test}
    normal = {col: 0 for col in columns_to_test}

    # Test normality for each column
    for column in columns_to_test:
        data = df[column].values
        # print(f"Data: {data}")
        # Ensure data used is at least 3 values
        if len(data) >= 3:
            stat, p = shapiro(data)
            # Use threshold to determine if the p-value is considered not normal distribution
            if p < 0.01:
                non_normal[column] += 1
                print(f"Not normal distribution for column: {column}")
            else:
                normal[column] += 1
            # Print stastic and p-value
            print(f"Statistic (Shapiro-Wilk test): {stat}, p-value: {p}")
        else:
            print(f"Not enough data points for normality test in column: {column}")
