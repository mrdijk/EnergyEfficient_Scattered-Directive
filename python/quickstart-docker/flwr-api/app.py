from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)


def run_subroutine(command):
    # Lists to capture the output and error lines
    output_lines = []
    error_lines = []

    # Start the subroutine
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,       # Ensure output is in text mode (not bytes)
        shell=True       # Required if the command is a string
    )

    # Stream output as it's generated and store it
    for line in process.stdout:
        print(line, end="")      # Stream to terminal
        output_lines.append(line)  # Store in the list

    # Stream error as it's generated and store it
    for line in process.stderr:
        print(line, end="")      # Stream error to terminal
        error_lines.append(line)  # Store in the list

    # Wait for the process to complete and get the exit status
    status = process.wait()

    # Join lists to return a single string for each
    output = ''.join(output_lines)
    error = ''.join(error_lines)

    # Return output, error, and status
    return output, error, status


@app.route('/run-script', methods=['POST'])
def run_script():
    print("Running script...")
    # # Execute a script from a file, e.g., "script.sh" or "script.py"
    # result = run_subroutine(["bash flwr-start.sh"], capture_output=True, text=True)
    output, error, status = run_subroutine("bash flwr-start.sh")

    # print(f"error:{result.stderr}")
    # print(f"output:{result.stdout}")
    # print(f"status:{result.returncode}")

    print(f"error:{error}")
    print(f"output:{output}")
    print(f"status:{status}")

    # Return the script output as the response
    # return jsonify({"output": result.stdout, "error": result.stderr, "status": result.returncode})
    return jsonify({"output": output, "error": error, "status": status})


@app.route('/test', methods=['POST'])
def run_test():
    print("in test...")
    # # Execute a script from a file, e.g., "script.sh" or "script.py"
    # result = run_subroutine(["bash flwr-start.sh"], capture_output=True, text=True)
    # output, error, status = run_subroutine("bash flwr-start.sh")

    # print(f"error:{result.stderr}")
    # print(f"output:{result.stdout}")
    # print(f"status:{result.returncode}")

    # print(f"error:{error}")
    # print(f"output:{output}")
    # print(f"status:{status}")

    # Return the script output as the response
    # return jsonify({"output": result.stdout, "error": result.stderr, "status": result.returncode})
    return jsonify({"status": "test ok"})


@app.route('/', methods=['POST'])
def run_root():
    print("in root...")
    # # Execute a script from a file, e.g., "script.sh" or "script.py"
    # result = run_subroutine(["bash flwr-start.sh"], capture_output=True, text=True)
    # output, error, status = run_subroutine("bash flwr-start.sh")

    # print(f"error:{result.stderr}")
    # print(f"output:{result.stdout}")
    # print(f"status:{result.returncode}")

    # print(f"error:{error}")
    # print(f"output:{output}")
    # print(f"status:{status}")

    # Return the script output as the response
    # return jsonify({"output": result.stdout, "error": result.stderr, "status": result.returncode})
    return jsonify({"status": "root ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
