# Current TODOs

- Get all the runtime information
- Add runtime controls beyond learning rate such as dropout rates, optimizer momentum, and batch size adjustments. These should be delivered via JetStream messages similar to learning rate updates.
- Implement PyTorch Lightning integration by creating a callback that listens to JetStream and applies runtime changes at the end of each epoch. Document how a context manager alternative could manage the JetStream connection around `Trainer.fit()`.
- Provide examples showing how to register devices and handle failure events using the new runtime controls.
