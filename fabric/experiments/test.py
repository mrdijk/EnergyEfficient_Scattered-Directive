import os
from datetime import datetime

timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

clients = [3,5]
cycles = [5,10]
for client in clients:
    for cycle in cycles:
        output_dir = os.path.join('data', f'{client}', f'{cycle}', f'{timestamp}')
        print(output_dir)
        # os.makedirs(output_dir, exist_ok=True)

print(dict(zip(["client1", "client2", "client3", "client4", "client5", "client6", "client7", "client8", "client9", "client10", "client11", "client12", "client13", "client14","client15","client16","client17","client18","client19", "client20"],
[3799, 10570, 4725, 2182, 17938, 2447, 1681, 1729, 6896, 14812, 2778, 3746, 4337, 2146, 2665, 1711, 2094, 3188, 2265, 8281])))