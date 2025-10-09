import json
import matplotlib.pyplot as plt

def plot(path):
    # Load the JSON results file
    with open(path, 'r') as f:
        data = json.load(f)
        
    results = data['results']

    # Extract training rounds, accuracies, and number of clients
    rounds = [entry['round'] for entry in results]
    server_accuracy = [entry['server_accuracy'] for entry in results]
    # Take the average accuracy across clients for each round
    client_accuracies = [sum(entry['client_accuracies']) / len(entry['client_accuracies']) 
                         for entry in results]
    clients = [entry['num_clients'] for entry in results]

    # Plot accuracy over rounds, color by number of clients
    plt.figure(figsize=(10, 6))

    # Find transition points where number of clients changes
    segments = []
    start_idx = 0
    for i in range(1, len(clients)):
        if clients[i] != clients[i-1]:
            segments.append((start_idx, i, clients[i-1]))
            start_idx = i
    segments.append((start_idx, len(clients), clients[-1]))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, (start, end, n_clients) in enumerate(segments):
        plt.plot(rounds[start:end], client_accuracies[start:end], 
                 label=f"{n_clients} clients", color=colors[idx % len(colors)])

    # Plot server accuracy in solid black line
    plt.plot(rounds, server_accuracy, label="Server Accuracy", color='black', linewidth=2)

    plt.xlabel('Training Round')
    plt.ylabel('Average Client Accuracy (%)')
    plt.title('Training Accuracy Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('vfl_accuracy_plot.png')
    plt.show()