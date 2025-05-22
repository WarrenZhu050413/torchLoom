# Hith Priority TODOs

## UI and Monitoring Enhancements

- Integrate https://gitee.com/gdtongji/torchft-ui

1. Display comprehensive DR Mapping
2. Display all the display information in a single page
2. Show real-time training status for each replica
3. Maintain history of past failures
- Look at the UI examples in draw.io file.

Deliverable: UI. 

## Dynamic Configuration
1. Enable hyperparameter broadcasting from training runtime to controller
   - Maintain a map: replica_id → hyperparameters
   - Allow dynamic tuning of parameters when training becomes unstable
   - Enable real-time configuration changes without training interruption

Deliverable: MNIST training with configurable parameters through torchLoom.

2.  Dynamic dataloader modifications https://github.com/pytorch/torchft/issues/171
   - Add/remove transformations on-the-fly
   - Adjust batch sizes
   - Maintain constant global batch size when scaling
   - Automatic learning rate adjustments
   - Multi-processing support (tricky, but performance critical)

Deliverable: Enable dynamic dataloader modification from 2 Replica Group to 1 Replica Group.

## Integration

3. Create Loom as an extension to Pytorch.

Deliverable: MNIST training with configurable parameters through torchLoom.

4. Handle resource draining by properly deleting DR mapping entries from both NATS JetStream and controller map

## Other feaures
1. Implement "drain" functionality for GPUs requiring replacement


### Integration with External Systems
1. SuperBench integration for failure prediction
   - Use ML models to predict GPU failure probabilities
   - Automatically manage GPU testing and replacement cycle

### Performance Monitoring
1. Live straggler detection
   - Track allreduce group membership and timing
   - Identify and handle performance bottlenecks
   - Calculate straggler impact and selectively remove from training
   - Implement dynamic timeout adjustments

2. Add configurable maximum iteration support


# Low Priority TODOs

3. Integrate with checkpointing system
   - Allow controller to specify which checkpoint to load
   - Synchronize across distributed training processes

4. Make controller and Manager handle NATS connection failures (e.g. when we have a network partition or the server is down) gracefully, e.g. by retrying.

5. Implement persistence for DR mapping in an object store: https://docs.nats.io/nats-concepts/jetstream/obj_store/obj_walkthrough

6. Improving support for multiple jobs (may need to maintain a JobID on top of ReplicaID)