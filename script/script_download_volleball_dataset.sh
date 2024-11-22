#!/bin/bash

# Check if the Kaggle API is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI is not installed. Please install it with 'pip install kaggle'."
    exit 1
fi

### Commands ###
# chmod 600 .kaggle/kaggle.json 
# chmod +x download_volleball_dataset.sh
# ./download_volleball_dataset.sh

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the parent directory (one level up from the script's directory)
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Define the path for the 'data' directory
DATA_DIR="$PARENT_DIR/data"


if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
    echo "Created directory '$DATA_DIR'."
else
    echo "Directory '$DATA_DIR' already exists."
fi

# Download the dataset using the Kaggle API
echo "Downloading dataset..."
kaggle datasets download -d ahmedmohamed365/volleyball -p "$DATA_DIR" --unzip

# Confirm completion
if [ $? -eq 0 ]; then
    echo "Dataset downloaded and unpacked successfully into '$DATA_DIR'."
else
    echo "Failed to download the dataset."
    exit 1
fi
