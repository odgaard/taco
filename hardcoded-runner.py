import benchmark_runner

def benchmark_run(mat, benchmark, method="random"):
    # hard-coded values
    taco_image = "odgaard/bacobench:0.0.7"
    hypermapper_image = "odgaard/hypermapper-grpc:0.0.2"

    mat_path = f"{mat}/{mat}.mtx"
    json = f"build/experiments/outdata_{benchmark}_{mat}/{mat}/{method}/cpp_taco_{benchmark}_{method}/cpp_taco_{benchmark}_{method}_scenario.json"

    benchmark_runner.run_program(taco_image, mat_path, method, benchmark, hypermapper_image, json)


if __name__ == "__main__":
    benchmark_run("Goodwin_040", "SpMM")
