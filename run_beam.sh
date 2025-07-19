#!/bin/bash

# This script runs the beam_cli.py simulation for a user-defined range of years.
# It requires a start and end year to be passed as command-line arguments.

# Exit immediately if any command fails, preventing partial runs.
set -e

# --- Argument Validation ---
# Check if exactly two arguments were provided.
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <start_year> <end_year>"
    echo "Example: $0 2022 2024"
    exit 1
fi

# Assign the command-line arguments to variables for clarity.
START_YEAR=$1
END_YEAR=$2

echo "Starting batch simulation run for Week 1 from $START_YEAR to $END_YEAR..."

# Loop through the sequence of years from START_YEAR to END_YEAR.
# We use `seq` because it reliably handles variables in loops.
for year in $(seq "$START_YEAR" "$END_YEAR")
do
  # Define the output filename using the current year from the loop
  output_file="./results/beam_${year}_wk-1_k10000.csv"

  # Print a message to show the current progress
  echo "==> Running simulation for Year: $year | Output: $output_file"

  # Execute the python script with the current year and dynamic output filename.
  python simulation/beam_cli.py \
    --year "$year" \
    --week 1 \
    --k 10000 \
    --n 1 \
    --output "$output_file"

done

echo "Batch simulation run completed successfully for years $START_YEAR-$END_YEAR."