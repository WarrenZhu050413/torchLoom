# Plan for Weavelet Integration

## Goal
Integrate a weavelet process into the Lightning training example so it can receive configuration updates from the weaver. The weavelet should run in a separate process and communicate optimizer type changes to the training process. Callback hooks will check the weavelet queue and update the optimizer when required.

## Steps
- Create a new module `torchLoom/weavelet.py` with a simple async loop that connects to NATS and listens for config messages. When an `optimizer_type` value is found it is put on a multiprocessing queue.
- Update `train.py`:
  - Spawn the weavelet process before starting training.
  - Extend `LightningTransformer` with dynamic optimizer creation and an `update_optimizer` method.
  - Add `WeaveletCallback` that checks the queue at the start of each epoch and applies optimizer updates.
- Add unit tests verifying that `update_optimizer` and the callback correctly change the optimizer based on queued values.
- Run `pytest` to ensure the repository remains stable.
