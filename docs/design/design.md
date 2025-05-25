# TorchLoom

## 1. Why torchLoom?

Distributed Machine Learning (ML) systems have evolved beyond the static, SPMD-centric "device-mesh + static schedule" approach, requiring more dynamic control to manage heterogeneous workloads. Modern workloads, including semi-synchronous training, RLHF pipelines, Mixture-of-Experts (MoE), and heterogeneous accelerators, are inherently MPMD (Multiple Program, Multiple Data). The complexities in MPMP introduces considerations like handling complex fault tolerance requirements, making optimizations due to physical resource availability (e.g. network bandwidth), adapting synchronization strategies (e.g., LocalSGD), switching optimizers (e.g., Muon), implementing more efficient collective communications (e.g., PCCL), or dynamically adjusting the current training strategy based on runtime metrics. These lead to complex control flows that are difficult to program and understand. Currently, these control flows are programmed with the training process script, making it difficult to read, understand, or reason about.

Recent systems research points to the need for a controller in ML training, and are moving away from SPMD:
 * HybridFlow (2024): Moves all control logic to a single controller, simplifying RLHF pipeline programming.
 * Pathways (2022): Uses centralized scheduling to enable fine-grained task routing across thousands of devices.
 * PCCL (2025): Leverages runtime-tuned collective communication strategies for higher efficiency, uses a single Master node for network optimization and fault tolerance.
 * torchFT:

torchLoom is inspired by torchFT's Lighthouse structure, and comes from investigations and contributions to torchFT: https://github.com/pytorch/torchft/issues/188, https://github.com/pytorch/torchft/issues/171, https://github.com/pytorch/torchft/pull/202. torchFT's lighthouse introduces a passive Lighthouse for coordination of Data Parallel groups in distributed training for fault tolerance. However, our imagination was more and more captured when thinking about torchFT in considering the possibilities for an active controller that dynamically adjusts parameters, steers traffic, and reconfigures the training graph before nodes fail.

torchLoom is that active controller. At its core it is composed of two components, the Weaver controller service, and the Threadlet agent. The Weaver service acts as the central planner, dynamically coordinating the interactions and state of distributed workers to optimize performance, fault tolerance, and resource utilization in real-time. The Weaver service communicattes with a threadlet process spawned in each training process. The threadlet acts as a sidecar that manages logging, monitoring, I/O control, and configuration management.

This control is crucial for dynamic training configuration, infrastructure aware optimization, and advanced fault tolerance.

⸻
## 2. Design Goals
The key goals for torchLoom are:
 * MPMD-first: Express heterogeneous pipelines as first-class citizens.
 * Single Controller, Multiple Data: Centralize all decision-making in one Weaver service and keeping the training workers focused on data flow. This centralized control simplifies global coordination and provides a holistic system view, drawing inspiration from systems like Google's Pathways and Bytedance's Hybridflow.
 * Fault tolerance: Build on torchFT's fault-tolerance APIs but allow the controller to prevent downtime through live reconfiguration. This proactive approach, informed by research on failure prediction, aims to minimize downtime compared to reactive methods like checkpoint/restart.
 * Network-aware: Taking inspiration from work like PCCL, communication efficient optimizers (DeMo), low-communication allreduces (int8 Allreduce, Mirror Reduce), and network-aware training algorithms (DiLoCo), as runtime-mutable parameters to optimize network communication dynamically.
⸻
## 3. Typical Workflows Enabled
torchLoom enables several workflows that provide adaptability, infrastructure-aware optimization, proactive fault tolerance, and interactive debugging:
### 3.1 Dynamic Training Configuration
 * Semi-Synchronous Training Adjustments: In semi-synchronous training algorithms like LocalSGD or DiLoCo, torchLoom dynamically adjusts parameters such as:
   * Modifying synchronization frequency (e.g., sync_every) based on network conditions or loss curve convergence.
   * Changing learning rates or optimizers at runtime depending on observed instability or training progress.
   * Adjusting batch sizes interactively to mitigate the impact of stragglers or optimize resource utilization.
 * Optimizer Switching: Depending on network conditions or the training phase, torchLoom dynamically adjusts the optimizer in response to these factors, switching to more efficient optimizers (e.g., PowerSGD or Top-K SGD).
 * Adaptive Parameters: Beyond learning rate and batch size, torchLoom can dynamically adjust model-specific parameters like:
   * Gradient clipping thresholds.
   * Noise injection levels for regularization.
   * Even model architecture (e.e.g., progressively adding Transformer blocks).
### 3.2 Infrastructure-Aware Optimization
 * Dynamic load balancing and batch size assignment (e.g. [DynMO](https://export.arxiv.org/pdf/2505.14864))
 * Node failure detection and replacement (e.g. [Superbench](https://www.usenix.org/conference/atc24/presentation/xiong))
 * Fault Detection: torchLoom ingests telemetry data from existing monitoring systems like C4, Revisiting Reliability, SuperBench, and TRANSOM, which track system resources and detect early signs of failure. This comprehensive telemetry, inspired by ByteDance's MegaScale system, is crucial for proactive fault management.
 * Checkpointing and Recovery: Techniques like CheckFreq, Check-N-Run, and DeepFreeze adjust checkpointing frequency based on network conditions and system load, ensuring minimal downtime during failure. Gemini ensures multi-node checkpointing is compatible with varying network speeds. torchLoom will prioritize in-memory checkpointing, similar to Gemini, for faster recovery.
 * Fast Worker Replacement: Techniques such as Varuna and Hoplite enable quick worker replacement with minimal disruption, relying on network and resource awareness to dynamically reallocate resources across the infrastructure.
 * Adaptive Network-Aware Recovery: torchLoom's adaptive checkpointing strategies ensure recovery occurs only when network conditions are optimal, reducing recovery time during fault events and preventing wasteful resource usage.
### 3.3 Advanced Fault Tolerance
 * Dynamic Reconfiguration: When detecting worker failures, torchLoom coordinates changes in the parallelism structure (e.g., Data Parallelism, Pipeline Parallelism, Tensor Parallelism) to maintain training continuity. This dynamic reconfiguration, similar to Oobleck, allows torchLoom to adapt to changing resource availability.
 * Message Rerouting: In architectures like MoE, if a node hosting certain experts fails, torchLoom reroutes messages to healthy experts or replicas. Similarly, in pipeline parallelism, data flow is rerouted around failed stages.
 * Proactive Failure Prediction: torchLoom incorporates time-series models to predict GPU or worker failures based on telemetry data. By analyzing historical failure data, it preemptively migrates workloads or adjusts training parameters, moving beyond reactive fault tolerance to proactive failure management. This failure prediction, drawing from recent research, aims to minimize downtime and data loss.
 * Fault Tolerance Approaches:
   * Checkpointing: torchLoom integrates fault tolerance strategies that adapt to network bandwidth and infrastructure state, ensuring effective recovery without overwhelming the system.
   * Graceful Restarts: torchLoom implements graceful restart mechanisms, ensuring the system can recover from failures without interrupting the training process.
   * Per-Step Fault Tolerance: torchLoom goes further than PyTorch Lightning by integrating active recovery mechanisms that adjust training parameters and reconfigure the graph to prevent failures from propagating.
### 3.4 Interactive Exploration and Debugging
 * Interactive Debugging: torchLoom enables interactive exploration of the loss landscape via a web UI, where knobs are just NATS messages to the Weaver. This allows for dynamic, real-time control over training processes, effectively "driving" the training process.

⸻
## 4. Related Work
Several other systems provide features related to fault tolerance, dynamic training adjustments, and distributed ML control. Here's how torchLoom differentiates itself:
 * Torchrun: Provides robust restart mechanisms during distributed training. torchLoom implements graceful restart mechanisms and dynamically adjusts training parameters during recovery to prevent downtime.
 * TorchFT: Introduces per-step fault tolerance, allowing for reconfiguration of fault-tolerant process groups at runtime to recover from failures dynamically. torchLoom extends this by adding active control over the training graph and fault recovery, preventing failures before they occur.
 * PyTorch Lightning:
   * Past Approach: Previously required manual intervention to restart epochs, resulting in the loss of optimizer state.
   * Current Implementation: Tracks and restores sampler indices, random states, optimizers, learning rate schedulers, and callbacks to ensure training continuity. torchLoom improves upon this by actively adjusting training parameters and orchestrating fault recovery before failures surface, reducing downtime significantly.
 * TorchElastic: Provides semantics for specifying a minimum and maximum number of nodes along with the number of allowed restarts. torchLoom not only supports elasticity but also dynamically reconfigures training processes and workflows based on real-time infrastructure state.
 * Ray Tune + PBT (Population-Based Training): Integrates hyperparameter search using Ray Tune with PBT, allowing for dynamic adjustment of hyperparameters throughout training. torchLoom complements this by dynamically orchestrating training workflows, managing the interaction between workers, and optimizing resource usage based on real-time telemetry.
 * DeepSpeed: A library from Microsoft focused on optimizing large-scale distributed training. torchLoom adds to this by centralizing control, enabling real-time decisions based on the infrastructure's health, resource allocation, and performance needs.
 * Megatron (NVIDIA): Optimized for training large transformer models across multiple GPUs and nodes. torchLoom can dynamically reallocate resources for MoE models, efficiently adjusting training pipelines in response to worker failures and load.
⸻
## 5. Status & Next Steps
 * MVP: The minimum viable product will include central control of learning rate and synchronization parameters, a PCCL AllReduce tuner, and recovery features built on torchFT-based fault tolerance.
 * Near-term: Development will include a UI for interactive "driving" and a plug-in policy engine (either learned or rule-based).
 * Long-term: The vision includes publishing logged trajectories as datasets and training "Kevin-bots" using imitation learning to autosteer new runs, further enhancing the system's ability to optimize itself.
⸻
## 6. Conclusion

torchLoom distills lessons from torchFT, HybridFlow, and Pathways into a unified controller that treats distributed training as a dynamic, living system rather than a static script. By embracing MPMD and centralizing control, it transforms experimentation speed, reliability, and performance into first-class, programmable features, paving the way for more efficient and flexible distributed ML systems. The design emphasizes dynamic adaptation and proactive fault tolerance, recognizing the realities of large-scale ML deployments where failures are common and resource availability can change.
