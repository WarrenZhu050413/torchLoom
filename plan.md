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

## Issues Resolved

### Pyre Type Checker Error (Fixed)
**Problem**: The Pyre linter was incorrectly inferring that `opt_type` from `queue.get_nowait()` was a Tensor instead of a string, causing this error:
```
Object of type "Tensor" is not callable
Attribute "__call__" is unknown
```

**Root Cause**: The multiprocessing queue's `get_nowait()` method returns `Any` type, and Pyre couldn't infer that it should be a string in this context.

**Solution**: Added explicit type annotation in `train.py` line 69:
```python
opt_type: str = self.queue.get_nowait()
```

**Verification**: 
- Test passed showing queue returns string type correctly
- Related pytest tests continue to pass
- Code imports and runs without runtime errors

This fix ensures that Pyre understands the expected type while maintaining all existing functionality.
