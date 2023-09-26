apptainer exec --bind ./build/experiments:/app/build/experiments   --bind ./hypermapper_dev:/app/hypermapper_dev   ../bacobench-hypermapper_0.0.1.sif /app/hypermapper_dev/run.sh
