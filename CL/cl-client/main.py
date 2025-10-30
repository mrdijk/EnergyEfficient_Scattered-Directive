import pandas as pd
import os
import requests

if os.getenv('ENV') == 'PROD':
    import config_prod as config
else:
    import config_local as config

def load_data(file_path):
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()
    file_name = f"{file_path}/{DATA_STEWARD_NAME}Data.csv"

    if DATA_STEWARD_NAME == "":
        print("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
    except FileNotFoundError:
        print(f"CSV file for table {file_name} not found.")
        return None

    return data

def send_data(data):
    return data
            
def main():
    global config
    
    dataset = load_data(config.dataset_filepath)
    response = requests.post(config.server_url, dataset)
    print(response)

if __name__ == "__main__":
  main()
