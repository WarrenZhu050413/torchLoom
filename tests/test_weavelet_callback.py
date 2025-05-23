import queue
from unittest.mock import Mock, patch

import torch

from torchLoom.lightning_integration import weavelet_handler
from torchLoom.weavelet import Weavelet
from train import EnhancedLightningTransformer, LightningTransformer, WeaveletCallback


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

                # Test optimizer update directly - now using enhanced handler system
            initial_optimizer_type = model.optimizer_type
            assert initial_optimizer_type == "SGD"

                # Test the enhanced handler dispatch system
            model.update_optimizer("Adam")
            assert model.optimizer_type == "Adam"
            assert isinstance(trainer.optimizers[0], torch.optim.Adam)

            # Test end callback
            cb.on_train_end(trainer, model)


def test_enhanced_handler_system():
    """Test the new enhanced handler system with decorators and automatic dispatch."""
    with patch("torchLoom.utils.get_device_uuid", return_value="test-device-uuid"):
        with patch("nats.connect"):  # Mock NATS connection
            with patch("multiprocessing.Process") as mock_process:
                # Mock the process to prevent actual subprocess creation
                mock_process_instance = Mock()
                mock_process.return_value = mock_process_instance
                mock_process_instance.start.return_value = None
                mock_process_instance.is_alive.return_value = True
                mock_process_instance.pid = 12345

                # Test basic weavelet with handler registration
                weavelet = Weavelet(replica_id="test_replica")

                # Track state changes
                state = {"optimizer": "SGD", "lr": 0.01, "enabled": True}

                # Register handlers using decorators
                @weavelet.handler("optimizer_type")
                def update_optimizer(new_type: str):
                    state["optimizer"] = new_type

                @weavelet.handler("learning_rate")
                def update_lr(new_lr: float):
                    state["lr"] = new_lr

                @weavelet.handler("enabled")
                def update_enabled(enabled: bool):
                    state["enabled"] = enabled

                # Test handler registration
                assert "optimizer_type" in weavelet._handlers
                assert "learning_rate" in weavelet._handlers
                assert "enabled" in weavelet._handlers

                # Test type information stored
                assert weavelet._handler_types["optimizer_type"] == str
                assert weavelet._handler_types["learning_rate"] == float
                assert weavelet._handler_types["enabled"] == bool

                # Test automatic handler dispatch
                weavelet._dispatch_handlers({"optimizer_type": "Adam"})
                assert state["optimizer"] == "Adam"

                weavelet._dispatch_handlers({"learning_rate": 0.001})
                assert state["lr"] == 0.001

                weavelet._dispatch_handlers({"enabled": False})
                assert state["enabled"] == False

                # Test multiple simultaneous updates
                weavelet._dispatch_handlers(
                    {"optimizer_type": "RMSprop", "learning_rate": 0.1, "enabled": True}
                )
                assert state["optimizer"] == "RMSprop"
                assert state["lr"] == 0.1
                assert state["enabled"] == True


def test_type_validation():
    """Test type validation and conversion in the enhanced handler system."""
    with patch("multiprocessing.Process") as mock_process:
        mock_process_instance = Mock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.start.return_value = None
        mock_process_instance.is_alive.return_value = True

        weavelet = Weavelet(replica_id="test_validation")

        # Test type conversion
        result_float = weavelet._validate_and_convert_value("test", "1.5")
        # Since no handler is registered, it returns as-is
        assert result_float == "1.5"

        # Register a handler with type information
        @weavelet.handler("learning_rate")
        def update_lr(new_lr: float):
            pass

        # Test float conversion
        result = weavelet._validate_and_convert_value("learning_rate", "0.001")
        assert result == 0.001
        assert isinstance(result, float)

        # Test int to float conversion
        result = weavelet._validate_and_convert_value("learning_rate", 1)
        assert result == 1.0
        assert isinstance(result, float)

        # Test boolean conversion
        @weavelet.handler("enabled")
        def update_enabled(enabled: bool):
            pass

        # Test various boolean conversions
        assert weavelet._validate_and_convert_value("enabled", "true") == True
        assert weavelet._validate_and_convert_value("enabled", "false") == False
        assert weavelet._validate_and_convert_value("enabled", "1") == True
        assert weavelet._validate_and_convert_value("enabled", "0") == False
        assert weavelet._validate_and_convert_value("enabled", 1) == True
        assert weavelet._validate_and_convert_value("enabled", 0) == False

        # Test invalid conversion
        try:
            weavelet._validate_and_convert_value("learning_rate", "invalid_float")
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Cannot convert value" in str(e)
            assert "invalid_float" in str(e)


def test_enhanced_lightning_integration():
    """Test the enhanced Lightning integration with automatic handler registration."""
    with patch("torchLoom.utils.get_device_uuid", return_value="test-device-uuid"):
        with patch("nats.connect"):  # Mock NATS connection
            with patch("multiprocessing.Process") as mock_process:
                # Mock the process
                mock_process_instance = Mock()
                mock_process.return_value = mock_process_instance
                mock_process_instance.start.return_value = None
                mock_process_instance.is_alive.return_value = True
                mock_process_instance.pid = 12345

                model = EnhancedLightningTransformer(
                    vocab_size=10, replica_id="enhanced_test"
                )

                # Test that handlers are automatically registered
                assert "optimizer_type" in model.weavelet._handlers
                assert "learning_rate" in model.weavelet._handlers

                # Test handler functionality
                initial_optimizer = model.optimizer_type
                assert initial_optimizer == "SGD"

                # Test automatic handler dispatch
                model.weavelet._dispatch_handlers({"optimizer_type": "Adam"})
                assert model.optimizer_type == "Adam"

                # Test learning rate update
                initial_lr = model.lr
                model.weavelet._dispatch_handlers({"learning_rate": 0.001})
                assert model.lr == 0.001


def test_weavelet_config_handler_backward_compatibility():
    """Test backward compatibility with the old config handler approach."""
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

                # Test the config handler directly - handlers are automatically called now
            assert model.optimizer_type == "SGD"

                # Test config update via the enhanced system (automatic dispatch)
                config_update = {"optimizer_type": "Adam"}
                model.weavelet._dispatch_handlers(config_update)
            assert model.optimizer_type == "Adam"
            assert isinstance(trainer.optimizers[0], torch.optim.Adam)

                # Test multiple config parameters including learning rate
                config_update = {"optimizer_type": "SGD", "learning_rate": 0.001}
                model.weavelet._dispatch_handlers(config_update)
                assert model.optimizer_type == "SGD"
                assert model.lr == 0.001
                assert isinstance(trainer.optimizers[0], torch.optim.SGD)


def test_auto_dispatch_toggle():
    """Test enabling/disabling automatic handler dispatch."""
    with patch("multiprocessing.Process") as mock_process:
        mock_process_instance = Mock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.start.return_value = None
        mock_process_instance.is_alive.return_value = True

        weavelet = Weavelet(replica_id="test_toggle")

        # Test auto dispatch is enabled by default
        assert weavelet._auto_dispatch == True

        # Test disabling auto dispatch
        weavelet.disable_auto_dispatch()
        assert weavelet._auto_dispatch == False

        # Test enabling auto dispatch
        weavelet.enable_auto_dispatch()
        assert weavelet._auto_dispatch == True
