# torchLoom

torchLoom is a runtime monitoring and control system for distributed AI training workloads. Named after the Mesopotamian god who governed the universe, torchLoom provides comprehensive monitoring and dynamic control capabilities for PyTorch training jobs.

It is inspired by [torchFT](https://github.com/pytorch/torchft), an experimental framework for fault tolerance distributed training in PyTorch. torchFT introduces the concept of a `Manager` within each training process and a `Lighthouse` that orchestrates failure detection and failure recovery through run-time process group reconfiguration, live checkpoint recovery, and failure detection. In contrast to torchFT, which coordiantes distributed quorum algorithm and has a relatively passive role, torchLoom is much more aggressive in providing new functionalities to the original traaining process.

## Overview

Unlike traditional orchestration frameworks that focus on deployment and scheduling, torchLoom provides run-time monitoring and control of the training process itself. It enables:

- **Real-time failure detection** and recovery for GPUs and training replicas
- **Dynamic configuration adjustment** without stopping training
- **Resource mapping** between hardware devices and training replicas

## Key Components

- **Weaver**: Central service that maintains device-to-replica mappings and coordinates failure responses
- **Manager Integration**: Direct integration with TorchFT Manager for seamless fault-tolerance
- **NATS Messaging**: Lightweight pub/sub communication layer for system events
- **Monitor CLI**: Command-line interface for manual control and testing

## Environment Setup

If you are in an Ubuntu based system, you can directly run the following command to set up the environment:

```sh
chmod +x setup.sh
./setup.sh
```

If you are in a Mac system, you can directly run the following command to set up the environment:

```sh
conda env create -f environment.yaml

brew install nats-io/nats-tools/nats-tools
./setup.sh
```

## Quick Start

0. Set up and clean up

```bash
cd /path/to/torchLoom
conda activate nats-torch27
```

1. Start the nats server
```
./nats/nats-server -c ./nats/nats.conf
```

2. Start the weaver
```sh
python -m torchLoom.weaver
```

3. Start the cli

```sh
python -m torchLoom.monitor_cli
```

4. Start the training script

```sh
python examples/pytorch/mnist.py
```

Then, you can go to [tutorial.md](tutorial.md) for detailed testing scenarios and instructions to see what torchLoom is capable of.

## Documentation

- [Design](docs/design/design.md): System architecture and concepts
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