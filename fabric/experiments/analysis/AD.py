import os

import numpy as np
import pandas as pd
from pycaret.anomaly import *
from scipy.spatial import distance
from sklearn import preprocessing

# Import BIRCH libraries (sklearn comes from "pip install scikit-learn")
from sklearn.cluster import Birch

# Import utils
from utils import get_folder_path, unix_time_to_datetime

# Folder where the gathered energy metrics are stored
# ENERGY_DATA_FOLDER = get_folder_path('data-collection/data', 1)
ENERGY_DATA_FOLDER = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1"
# Path to the energy metrics file (from get_metrics.py)
ENERGY_DATA_FILE = os.path.join(ENERGY_DATA_FOLDER, 'combined_energy_stats.csv')
# Folder where the algorithms data is stored
ALGORITHMS_DATA_FOLDER = get_folder_path('data/exp1')
# Path to the converted energy metrics file (after CPU conversion)
CONVERTED_ENERGY_DATA_FILE = os.path.join(
    ALGORITHMS_DATA_FOLDER, 'converted_energy_metrics.csv')


def main():
    """
    Main function of the anomaly detection.
    """
    # Convert the CPU metrics to percentage
    # cpu_to_percentage()

    # Training not required for BIRCH here, it is done in the BIRCH function itself

    """
    Ground truth in the study from Ivano and his colleagues refers to the labeled data that indicates which data points
    are actual anomalies, generated using statistical thresholds. It is used as a benchmark to evaluate the performance
    of the anomaly detection models, allowing comparison between the model's predictions and the known anomalies.

    However, for DYNAMOS, only the output of the algorithms is required, so the ground truth is not used in this case
    and therefore not included in the code.
    """

    # Execute BIRCH AD algorithm
    execute_BIRCH_AD_algorithm()


def cpu_to_percentage() -> None:
    print("Converting CPU metrics to percentage...")

    # Read the CSV file and parse the 'time' column while converting Unix timestamps
    result_df = pd.read_csv(ENERGY_DATA_FILE, parse_dates=[
                            "time"], date_parser=unix_time_to_datetime)

    # Additional processing remains the same
    temp_df = pd.read_csv(ENERGY_DATA_FILE)

    # Convert the cpu metrics to percentage
    for col in result_df.columns:
        if col.endswith("_cpu"):
            # Multiply the value by 100 to get the cpu percentage
            temp_df[col] = result_df[col] * 100

    # Remove disk data due to the existing bug in Cadvisor
    columns_to_drop = [col for col in temp_df.columns if '_disk' in col]
    temp_df.drop(columns=columns_to_drop, inplace=True)

    # Save the converted file in the data folder
    output_file = os.path.join(
        ALGORITHMS_DATA_FOLDER, 'converted_energy_metrics.csv')
    print(f"Saving the converted data to {output_file}...")
    # Write the data to the output file
    temp_df.to_csv(output_file, index=False)


def run_BIRCH_AD_with_smoothing(temp_df: pd.DataFrame, df: pd.DataFrame, column) -> pd.DataFrame:
    # Define anomaly detection threshold for BIRCH clustering
    ad_threshold = 0.045
    # Define window size for smoothing the data
    smoothing_window = 12
    # Extract the time and the specific column for analysis
    test_df = df.loc[:, ["time", column]]

    # Iterate over each column in the test dataframe
    for column_name, column_data in test_df.items():
        # Skip the 'time' column since it's not relevant for clustering
        if column_name != 'time':
            # Apply a rolling mean to smooth the column data (energy data e.g. CPU, memory, energy, etc.)
            column_data = column_data.rolling(
                window=smoothing_window, min_periods=1).mean()

            # Convert the smoothed series to a NumPy array
            x = np.array(column_data)
            # Replace NaN values with 0 to ensure no missing data is passed to the algorithm
            x = np.where(np.isnan(x), 0, x)
            # Normalize the data to ensure all values are on a similar scale
            normalized_x = preprocessing.normalize([x])
            # Reshape the normalized data to a 2D array as required by BIRCH
            X = normalized_x.reshape(-1, 1)

            # Initialize the BIRCH model with the specified parameters
            birch = Birch(branching_factor=50, n_clusters=None,
                          threshold=ad_threshold, compute_labels=True)

            # Build the CF tree for the input data (i.e. perform clustering)
            birch.fit(X)
            # Use the fitted model to predict the cluster labels for the data
            birch.predict(X)

            # Calculate the Euclidean distances (straight-line distance between two points in a plane or space)
            # from each data point to all cluster centers. `X` contains the data points, and `birch.subcluster_centers_`
            # contains the centers of the subclusters identified by BIRCH. `distance.cdist` computes the distance between
            # each data point and each cluster center.
            distances = distance.cdist(X, birch.subcluster_centers_)
            # For each data point, find the minimum distance to any of the cluster centers.
            # This minimum distance represents how close the point is to the nearest cluster center.
            # A smaller distance indicates that the point is well represented by the cluster, while a
            # larger distance might suggest that the point is an outlier or anomaly.
            min_distances = np.min(distances, axis=1)

            # Set a threshold to identify anomalies based on the distribution of minimum distances.
            # Here, the 95th percentile is chosen, meaning that points with a distance greater than 95%
            # of all points are considered anomalies. This is based on the assumption that most data points
            # should be near a cluster center, and only a few should be far away.
            threshold = np.percentile(min_distances, 95)

            # Label data points as anomalies (1) if their distance exceeds the threshold, else normal (0)
            test_df['anomaly_label'] = np.where(
                min_distances > threshold, 1, 0)

            # Update the original dataframe with the anomaly labels and scores as new columns
            temp_df = temp_df.assign(
                **{
                    # Anomaly labels (0 or 1)
                    f"{column}_Anomaly": test_df['anomaly_label'],
                    # Anomaly scores (distances)
                    f"{column}_Anomaly_Score": min_distances,
                }
            )

    # Return the updated dataframe with the anomaly information
    return temp_df


def execute_BIRCH_AD_algorithm():
    print("Start executing BIRCH AD algorithm...")
    # Read the converted energy metrics file
    df = pd.read_csv(ENERGY_DATA_FILE)
    temp_df = pd.read_csv(ENERGY_DATA_FILE)

    # Iterate over each column in the dataframe
    for column in df.columns:
        # Run BIRCH algorithm and accumulate the results. Update the temp_df with
        # the new results for this column (next iteration will use this updated data)
        temp_df = run_BIRCH_AD_with_smoothing(temp_df, df, column)

    # Save the combined results to the CSV file
    # temp_df contains the results for all columns
    results_path = os.path.join(ALGORITHMS_DATA_FOLDER, 'BIRCH_AD_results.csv')
    print(f"Saving BIRCH AD results data to {results_path}...")
    temp_df.to_csv(results_path, index=False)
