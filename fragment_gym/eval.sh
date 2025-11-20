#!/bin/bash

# Check if --start argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 start,stop,[step],[parallel] <script.py> [args...]"
    exit 1
fi

# Extract values from the argument
IFS=',' read -r -a arg_values <<< "$1"
start="${arg_values[0]}"
stop="${arg_values[1]}"
step="${arg_values[2]:-10}" # Set default value for step to 10
parallel="${arg_values[3]:-8}" # Set default value for parallel to 8

# Check if <script.py> argument is provided
if [ -z "$2" ]; then
    echo "Usage: $0 start,stop,[step],[parallel] <script.py> [args...]"
    exit 1
fi

script_py="$2"
shift 2  # Remove the processed arguments

# Initialize global sequence array
fresco_range_separators=()

# Function that mimics numpy's arange without using local -n
function generate_sequence() {
    local start=$1
    local stop=$2
    local step=$3

    for ((i=start; i<stop; i+=step)); do
        fresco_range_separators+=($i)
    done
}

# Example usage:
# Clear the sequence array before using
fresco_range_separators=()

# Call the generate_sequence function with the provided range and step
generate_sequence $start $((stop + 1)) $step  # Increase the stop value by 1
# Get the length of the array
length=${#fresco_range_separators[@]}
echo "Generated sequence: ${fresco_range_separators[@]}"

# Set number of parallel executions:
tsp -S "$parallel"

# Loop through the array, including the last element
for (( i=0; i<length; i++ )); do
    # Current item in the list
    range_start=${fresco_range_separators[$i]}
    range_stop=$((range_start + step - 1))  # Set the range_stop accordingly

    # Start a Python process with the tuple as an argument
    tsp python3 "$script_py" -m="eval" -f="$range_start,$range_stop" "$@"
done
