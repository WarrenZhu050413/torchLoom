import queue
import torch
from train import LightningTransformer, WeaveletCallback

class DummyProcess:
    def __init__(self):
        self.alive = True
    def is_alive(self):
        return self.alive

def test_callback_updates_optimizer():
    q = queue.Queue()
    proc = DummyProcess()
    model = LightningTransformer(vocab_size=10)
    # Simulate trainer with an optimizer list
    class DummyTrainer:
        def __init__(self, opt):
            self.optimizers = [opt]
    trainer = DummyTrainer(model.configure_optimizers())
    model.trainer = trainer

    cb = WeaveletCallback(q, proc)
    q.put("Adam")
    cb.on_train_epoch_start(trainer, model)
    assert isinstance(trainer.optimizers[0], torch.optim.Adam)

