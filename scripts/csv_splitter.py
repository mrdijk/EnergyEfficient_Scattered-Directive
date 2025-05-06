import csv
import json
from contextlib import ExitStack


def filter_data(rows, columns, filter_columns):
    return [d for i, d in enumerate(rows) if columns[i] in filter_columns]


with open('./scripts/csv_splitter_configuration.json', 'r') as file:
    configuration = json.load(file)

with open(configuration["file"]) as data_file:
    dataReader = csv.reader(data_file, delimiter=configuration["delimiter"])
    columns = next(dataReader)

    with ExitStack() as stack:
        files = [(partition["columns"],
                  csv.writer(stack.enter_context(open(partition["name"], "w")),
                             delimiter=configuration["delimiter"]))
                 for partition in configuration["partitions"]]

        for (filter_columns, writer) in files:
            writer.writerow(filter_data(columns, columns, filter_columns))

        for row in dataReader:
            for (filter_columns, writer) in files:
                writer.writerow(filter_data(row,
                                            columns,
                                            filter_columns))
