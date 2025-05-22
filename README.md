# torchLoom

torchLoom is a runtime monitoring and control system for distributed AI training workloads. Named after the Mesopotamian god who governed the universe, torchLoom provides comprehensive monitoring and dynamic control capabilities for PyTorch training jobs.

It is inspired by [torchFT](https://github.com/pytorch/torchft), an experimental framework for fault tolerance distributed training in PyTorch. torchFT introduces the concept of a `Manager` within each training process and a `Lighthouse` that orchestrates failure detection and failure recovery through run-time process group reconfiguration, live checkpoint recovery, and failure detection. In contrast to torchFT, which coordiantes distributed quorum algorithm and has a relatively passive role, torchLoom is much more aggressive in providing new functionalities to the original traaining process.

## Overview

Unlike traditional orchestration frameworks that focus on deployment and scheduling, torchLoom provides run-time monitoring and control of the training process itself. It enables:

- **Real-time failure detection** and recovery for GPUs and training replicas
- **Dynamic configuration adjustment** without stopping training
- **Resource mapping** between hardware devices and training replicas

## Key Components

- **Controller**: Central service that maintains device-to-replica mappings and coordinates failure responses
- **Manager Integration**: Direct integration with TorchFT Manager for seamless fault-tolerance
- **NATS Messaging**: Lightweight pub/sub communication layer for system events
- **Monitor CLI**: Command-line interface for manual control and testing

## Quick Start

First, open 6 bash terminals, execute `conda deactivate` in each terminal (potentially twice) if you have multiple conda environments locked within each other. 

Then, you can set of the environment. 

0. Set up and clean up

```bash
cd /path/to/torchLoom # e.g. cd /srv/apps/torchLoom
conda activate /path/to/env # e.g. conda activate /srv/apps/danny/miniconda3/envs/warren/torchtitan
# source ./torchLoom/scripts/preamble.sh # Sets up the environment, compiles protobuf, and kills all the existing servers.
```

1. Start the nats server
```
./torchLoom/my_nats/nats-server -c ./torchLoom/my_nats/nats.conf
```

2. Start the controller
```sh
python -m torchLoom.controller
```

3. Start the cli

```sh
python -m torchLoom.monitor_cli_danny
```

<!-- 
4. Set up torchFT by starting the lighthouse

```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

5. Run the torchFT training script on one device

Start Device 1:
```sh
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2

CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 -- train_ddp.py
```

6. Optionally, to test multiple device failures, you can run the training script on another device.

Start Device 2:
```sh
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2

CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 -- train_ddp.py
```

7. Now, control C on any of the training processes. See what happens! Also relaunch the training script, and see what happens!

When the training processes run, you should be able to see the devices registering their device_uuid and replica_id.

```sh
Registered device: GPU-307a982d-bf2b-4cc3-64e3-aae456bf6a28 for replica_id: train_ddp_0:d5aa538f-3268-4f78-ae88-3afff894e629 # For replica 0
Registered device: GPU-307a982d-bf2b-4cc3-64e3-aae456bf6a28 for replica_id: train_ddp_1:164ecd9c-f806-4eef-8fd3-add20298ea20 # For replica 1
``` -->

Then, you can go to [tutorial.md](tutorial.md) for detailed testing scenarios and instructions to see what torchLoom is capable of.

## Documentation

- [Design](design.md): System architecture and concepts
- [Tutorial](tutorial.md): Testing procedures and example workflows
- [Debugging](debugging.md): Troubleshooting guide for common issues
- [TODO](todo.md): Upcoming features and development roadmap

## Environment setup

- Do `conda env create -n torchLoom -f environment.yaml`

Also need to download protobuf compiler and library

On Debian-based machines, run
```bash
sudo apt install protobuf-compiler libprotobuf-dev
```