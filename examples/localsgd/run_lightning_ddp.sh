#!/bin/bash

# Script to run LocalSGD Lightning training on 2 GPUs
# This script provides two methods: using torchrun and using environment variables

echo "=== LocalSGD PyTorch Lightning Multi-GPU Training ==="

# Method 1: Using torchrun (recommended)
echo "Method 1: Using torchrun"
echo "Command:"
echo "torchrun --nproc_per_node=2 --nnodes=1 train_localsgd_lightning.py"
echo ""

# Method 2: Using environment variables with python -m lightning run
echo "Method 2: Using PyTorch Lightning's built-in launcher"
echo "Command:"
echo "python -m lightning run model --num-processes=2 train_localsgd_lightning.py"
echo ""

# Method 3: Manual environment variables (if you want to run with python directly)
echo "Method 3: Manual environment variables"
echo "You would need to run this command on each process (not recommended for most users):"
echo ""
echo "# For rank 0:"
echo "export MASTER_ADDR=localhost"
echo "export MASTER_PORT=29500"
echo "export WORLD_SIZE=2"
echo "export LOCAL_RANK=0"
echo "export RANK=0"
echo "python train_localsgd_lightning.py"
echo ""
echo "# For rank 1 (in a separate terminal):"
echo "export MASTER_ADDR=localhost"
echo "export MASTER_PORT=29500"
echo "export WORLD_SIZE=2"
echo "export LOCAL_RANK=1"
echo "export RANK=1"
echo "python train_localsgd_lightning.py"
echo ""

# Choose method
echo "Choose a method to run:"
echo "1) torchrun (recommended)"
echo "2) Lightning launcher" 
echo "3) Manual setup"
echo "q) Quit"
read -p "Enter choice [1-3, q]: " choice

case $choice in
    1)
        echo "Running with torchrun..."
        torchrun --nproc_per_node=2 --nnodes=1 train_localsgd_lightning.py
        ;;
    2)
        echo "Running with Lightning launcher..."
        python -m lightning run model --num-processes=2 train_localsgd_lightning.py
        ;;
    3)
        echo "Manual setup selected. Please run the commands shown above in separate terminals."
        ;;
    q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac 