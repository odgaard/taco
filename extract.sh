#!/bin/bash

# Set the directory where the tar.gz files are located
folder="data/suitesparse"

# Change to the target folder
cd "$folder"

# Iterate over all tar.gz files in the folder
for file in *.tar.gz; do
  # Extract the tar.gz file
  tar -xf "$file"
done

echo "Extraction completed."

