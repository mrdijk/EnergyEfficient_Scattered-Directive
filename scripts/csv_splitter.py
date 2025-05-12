import csv
import json
import os
from contextlib import ExitStack


def filter_data(rows, columns, filter_columns):
    return [d for i, d in enumerate(rows) if columns[i] in filter_columns]


with open('./scripts/csv_splitter_configuration.json', 'r') as file:
    configuration = json.load(file)

with open(configuration["file"]) as data_file:
    dataReader = csv.reader(data_file, delimiter=configuration["delimiter"])
    columns = next(dataReader)

    for partition in configuration["partitions"]:
        os.makedirs(partition["outputDirectory"], exist_ok=True)

    print("Creating datasets...")
    with ExitStack() as stack:
        files = [(partition["columns"],
                  csv.writer(stack.enter_context(
                      open(partition["outputDirectory"] +
                           partition["name"] + ".csv", "w")),
            delimiter=configuration["delimiter"]))
            for partition in configuration["partitions"]]

        for (filter_columns, writer) in files:
            writer.writerow(filter_data(columns, columns, filter_columns))

        for row in dataReader:
            for (filter_columns, writer) in files:
                writer.writerow(filter_data(row,
                                            columns,
                                            filter_columns))

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
