import queue
from unittest.mock import Mock, patch

import torch

from train import LightningTransformer, WeaveletCallback


class DummyTrainer:
    def __init__(self, opt):
        self.optimizers = [opt]


def test_callback_updates_optimizer():
    """Test that the new WeaveletCallback works with the process-based approach."""
    # Create a model with weavelet
    with patch("torchLoom.utils.get_device_uuid", return_value="test-device-uuid"):
        with patch(
            "nats.connect"
        ):  # Mock NATS connection to avoid actual network calls
            with patch("multiprocessing.Process") as mock_process:
                # Mock the process to prevent actual subprocess creation
                mock_process_instance = Mock()
                mock_process.return_value = mock_process_instance
                mock_process_instance.start.return_value = None
                mock_process_instance.is_alive.return_value = True
                mock_process_instance.pid = 12345
                
                model = LightningTransformer(vocab_size=10, replica_id="test_replica")

                # Set up a dummy trainer
                trainer = DummyTrainer(model.configure_optimizers())
                model.trainer = trainer

                # Test the callback
                cb = WeaveletCallback()

                # Test start callback
                cb.on_train_start(trainer, model)
                assert hasattr(model, "weavelet")
                assert model.weavelet._replica_id == "test_replica"

                # Test optimizer update directly
                initial_optimizer_type = model.optimizer_type
                assert initial_optimizer_type == "SGD"

                # Simulate an optimizer change via config update
                model.handle_config_update({"optimizer_type": "Adam"})
                assert model.optimizer_type == "Adam"
                assert isinstance(trainer.optimizers[0], torch.optim.Adam)

                # Test end callback
                cb.on_train_end(trainer, model)


def test_weavelet_config_handler():
    """Test that weavelet config handlers work properly with the process-based approach."""
    with patch("torchLoom.utils.get_device_uuid", return_value="test-device-uuid"):
        with patch("nats.connect"):  # Mock NATS connection
            with patch("multiprocessing.Process") as mock_process:
                # Mock the process to prevent actual subprocess creation
                mock_process_instance = Mock()
                mock_process.return_value = mock_process_instance
                mock_process_instance.start.return_value = None
                mock_process_instance.is_alive.return_value = True
                mock_process_instance.pid = 12345
                
                model = LightningTransformer(vocab_size=10, replica_id="test_replica")

                # Set up a dummy trainer
                trainer = DummyTrainer(model.configure_optimizers())
                model.trainer = trainer

                # Test the config handler directly
                assert model.optimizer_type == "SGD"
                
                # Test config update via the new method
                config_update = {"optimizer_type": "Adam"}
                model.handle_config_update(config_update)
                assert model.optimizer_type == "Adam"
                assert isinstance(trainer.optimizers[0], torch.optim.Adam)

                # Test multiple config parameters
                config_update = {
                    "optimizer_type": "SGD",
                    "learning_rate": "0.01"  # This would need to be handled if implemented
                }
                model.handle_config_update(config_update)
                assert model.optimizer_type == "SGD"
                assert isinstance(trainer.optimizers[0], torch.optim.SGD)
