from google.protobuf.struct_pb2 import Struct
import json
import os
import io
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd


def filter_data(rows, columns, filter_columns):
    return [d for i, d in enumerate(rows) if columns[i] in filter_columns]


# def _bin_age(age_series):
#     bins = [-np.inf, 18, 60, np.inf]
#     labels = ["Child", "Adult", "Elderly"]
#     return (
#         pd.cut(age_series, bins=bins, labels=labels, right=True)
#         .astype(str)
#         .replace("nan", "Unknown")
#     )


def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df):
    # Convert 'Age' to numeric, coercing errors to NaN
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    # df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    # keywords = set(df.columns)
    # print(all_keywords)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin"]
    )
    # print(df)
    return df


with open('./scripts/csv_splitter_configuration.json', 'r') as file:
    configuration = json.load(file)

df = pd.read_csv(configuration["file"])
processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
data = _create_features(processed_df)

for partition in configuration["partitions"]:
    os.makedirs(partition["outputDirectory"], exist_ok=True)
    print(partition["columns"])
    partition_data = data[list({
        column
        for column in data.columns
        for keyword in partition["columns"]
        if keyword in column
    })]

    partition_data.to_csv(
        partition["outputDirectory"] + partition["name"] + ".csv")

    # with open(configuration["file"]) as data_file:
    #     dataReader = csv.reader(data_file, delimiter=configuration["delimiter"])
    #     columns = next(dataReader)
    #
    #     for partition in configuration["partitions"]:
    #         os.makedirs(partition["outputDirectory"], exist_ok=True)
    #
    #     print("Creating datasets...")
    #     with ExitStack() as stack:
    #         files = [(partition["columns"],
    #                   csv.writer(stack.enter_context(
    #                       open(partition["outputDirectory"] +
    #                            partition["name"] + ".csv", "w")),
    #             delimiter=configuration["delimiter"]))
    #             for partition in configuration["partitions"]]
    #
    #         for (filter_columns, writer) in files:
    #             writer.writerow(filter_data(columns, columns, filter_columns))
    #
    #         for row in dataReader:
    #             for (filter_columns, writer) in files:
    #                 writer.writerow(filter_data(row,
    #                                             columns,
    #                                             filter_columns))

    # with open('./configuration/etcd_launch_files/datasets.json', 'r') as file:
    #     current_datasets = json.load(file)
    #
    #     datasets = []
    #     dataset_names = []
    #
    #     for partition in configuration["partitions"]:
    #         dataset_names += [partition["name"]]
    #         datasets += [{
    #             "name": partition["name"],
    #             "type": "csv",
    #             "delimiter": configuration["delimiter"],
    #             "tables": [partition["name"]],
    #             "sensitive_columns": {
    #                     partition["name"]: filter_data(partition["columns"],
    #                                                    partition["columns"],
    #                                                    configuration["sensitive"])
    #             }
    #         }]
    #
    #     for dataset in current_datasets:
    #         if dataset["name"] not in dataset_names:
    #             datasets += dataset
    #
    # with open('./configuration/etcd_launch_files/datasets.json', 'w') as file:
    #     json.dump(datasets, file)


# data = pd.read_csv('./python/vfl-train/datasets/alphaData.csv')[1:20]
#
#
# class ClientModel(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.fc = nn.Linear(input_size, 4)
#
#     def forward(self, x):
#         return self.fc(x)
#
#
# def train_model(data, learning_rate):
#     data = torch.tensor(StandardScaler().fit_transform(data)).float()
#     model = ClientModel(data.shape[1])
#     # optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#     embedding = model(data)
#     return embedding.detach().numpy()
#
#
# def serialise_array(array):
#     return json.dumps([
#         str(array.dtype),
#         array.tobytes().decode("latin1"),
#         array.shape])
#
#
# def deserialise_array(string):
#     encoded_data = json.loads(string)
#     dataType = np.dtype(encoded_data[0])
#     dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)
#
#     if len(encoded_data) > 2:
#
#         return dataArray.reshape(encoded_data[2])
#     return dataArray
#
#
# np.set_printoptions(threshold=sys.maxsize)
#
# embeddings = train_model(data, 0.05)
# print(type(embeddings))
# data = Struct()
# data.update({"embeddings": serialise_array(embeddings)})
#
# print(embeddings.shape)
# print(serialise_array(embeddings))
# print(deserialise_array(serialise_array(embeddings)))
# print(deserialise_array(serialise_array(embeddings)) == embeddings)


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


def serialise_array(array):
    return json.dumps([
        str(array.dtype),
        array.tobytes().decode("latin1"),
        array.shape])


def deserialise_array(string, hook=None):
    encoded_data = json.loads(string, object_pairs_hook=hook)
    dataType = np.dtype(encoded_data[0])
    dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)

    if len(encoded_data) > 2:
        return dataArray.reshape(encoded_data[2])

    return dataArray


def train_model(data, model):
    embedding = model(data)
    return embedding.detach().numpy()


def vfl_evaluate(data, model, optimiser, gradients):
    print("Start vfl_evaluate")

    try:
        model.zero_grad()
        embedding = model(data)
        embedding.backward(torch.from_numpy(gradients))
        optimiser.step()
    except Exception as e:
        print(f"Error occurred: {e}")


# Note: Gradients sent by server are for this client only to preserve privacy
def vfl_train(learning_rate, model_state, gradients):
    global config

    try:
        data = pd.read_csv('./python/vfl-train/datasets/alphaData.csv')[1:20]
    except Exception as e:
        print(f"Error occurred: {e}")

        # If data does not exist, shut down service
        print("Shutting down the service")
        return None, None

    data = torch.tensor(StandardScaler().fit_transform(data)).float()
    model = ClientModel(data.shape[1])

    if model_state is not None:
        print(model_state)
        model.load_state_dict(torch.load(
            io.BytesIO(model_state.encode("latin1"))))

    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if gradients is not None:
        vfl_evaluate(data, model, optimiser, gradients)

    embeddings = train_model(data, model)
    model_state = model.state_dict()

    buffer = io.BytesIO()
    torch.save(model_state, buffer)

    data = Struct()
    data.update({"embeddings": serialise_array(embeddings),
                 "model_state": buffer.getvalue().decode("latin1")})

    return data


vfl_train(0.05, None, None)
