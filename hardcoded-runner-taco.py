import benchmark_runner

def benchmark_run(mat, benchmark, method="random"):
    # hard-coded values
    runtime = "docker"
    taco_image = "odgaard/bacobench:0.0.7"
    hypermapper_image = "odgaard/hypermapper-grpc:0.0.2"

    json_base_path = f"build/experiments/outdata_{benchmark}_{mat}"
    json_end_path = f"cpp_taco_{benchmark}_{method}/cpp_taco_{benchmark}_{method}_scenario.json"

    if benchmark in ("MTTKRP", "TTV"):
        mat_path = f"{mat}.tns"
        json = f"{json_base_path}/{method}/{json_end_path}"
    else:
        mat_path = f"{mat}/{mat}.mtx"
        json = f"{json_base_path}/{mat}/{method}/{json_end_path}"

    benchmark_runner.run_program(runtime, mat_path, method, benchmark, taco_image, hypermapper_image, json, ("taco"))

if __name__ == "__main__":
    benchmark_run("Goodwin_040", "SpMM")
    #benchmark_run("Goodwin_040", "SDDMM")
    #benchmark_run("Goodwin_040", "SpMV")
    #benchmark_run("uber-pickups", "MTTKRP")
    #benchmark_run("uber3", "TTV")
