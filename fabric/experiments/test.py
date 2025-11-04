import pandas as pd
from google.protobuf.struct_pb2 import Struct, Value, ListValue

def dataframe_to_protobuf(df):
    # Convert the DataFrame to a dictionary of lists (one for each column)
    data_dict = df.to_dict(orient='list')

    # Convert the dictionary to a Struct
    data_struct = Struct()

    # Iterate over the dictionary and add each value to the Struct
    for key, values in data_dict.items():
        # Pack each item of the list into a Value object
        value_list = [Value(string_value=str(item)) for item in values]
        # Pack these Value objects into a ListValue
        list_value = ListValue(values=value_list)
        # Add the ListValue to the Struct
        data_struct.fields[key].CopyFrom(Value(list_value=list_value))

    # Create the metadata
    # Infer the data types of each column
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    # Convert the data types to string values
    metadata = {k: str(v) for k, v in data_types.items()}

    return data_struct, metadata

if __name__ == "__main__":
    test_csv = '/home/maurits/EnergyEfficiencient_FL/energy-efficiency/experiments/data/baseline_ComputeToData_250124-1654/exp_0/full_experiment_results.csv'
    result = pd.read_csv(test_csv)
    data, metadata = dataframe_to_protobuf(result)
   
    print('data: ', data)
    print('metadata: ', metadata)