import os
import docker
import argparse
import concurrent.futures

def run_docker_with_args_taco(image_name, mat_val, method_val, o_val):
    client = docker.from_env()

    command = [
        '-mat', mat_val,
        '--method', method_val,
        '-o', o_val
    ]

    # Define the volume mapping
    volume_mapping = {
        os.path.join(os.getcwd(), 'build/experiments'): {
            'bind': '/app/build/experiments',
            'mode': 'rw'
        }
    }
    container = client.containers.run(
            image_name,
            command,
            detach=True,
            network_mode="host",
            volumes=volume_mapping
    )
    return container

def run_docker_with_args_hypermapper(image_name, json_file):
    client = docker.from_env()
    command = [json_file]

    volume_mapping = {
        os.path.join(os.getcwd(), 'build/experiments'): {
            'bind': '/app/build/experiments',
            'mode': 'rw'
        }
    }
    container = client.containers.run(
            image_name,
            command,
            detach=True,
            network_mode="host",
            volumes=volume_mapping
    )
    return container

def collect_logs(container, name):
    for line in container.logs(stream=True):
        text = line.strip().decode('utf-8')
        print(f"[{name}]: {text}")

def run_program(taco_image, mat, method, o, hypermapper_image, json):
    taco_container = None
    hypermapper_container = None

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            taco_future = executor.submit(run_docker_with_args_taco, taco_image, mat, method, o)
            hypermapper_future = executor.submit(run_docker_with_args_hypermapper, hypermapper_image, json)

        taco_container = taco_future.result()
        hypermapper_container = hypermapper_future.result()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(collect_logs, taco_container, "TACO")
            executor.submit(collect_logs, hypermapper_container, "Hypermapper")

    except (SystemExit, KeyboardInterrupt):
        print("Interrupt detected. Stopping containers...")
    except Exception as e:
        print(f"Error: {e}. Stopping containers...")
    finally:
        if hypermapper_container:
            hypermapper_container.stop()
        if taco_container:
            taco_container.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Docker images with specific commands.")

    parser.add_argument("--taco-image", required=True, help="Docker image name for the taco command.")
    parser.add_argument("--mat", required=True, help="Argument for '-mat' in the taco command.")
    parser.add_argument("--method", default="random", help="Argument for '--method' in the taco command.")
    parser.add_argument("-o", required=True, help="Argument for '-o' in the taco command.")

    parser.add_argument("--hypermapper-image", required=True, help="Docker image name for the hypermapper command.")
    parser.add_argument("--json", required=True, help="JSON file argument for the hypermapper command.")

    args = parser.parse_args()

    run_program(args.taco_image, args.mat, args.method, args.o, args.hypermapper_image, args.json)

