import argparse
import asyncio
import cmd

import nats
from torchLoom.config import Config
from torchLoom.torchLoom_pb2 import EventEnvelope
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="torchLoom_monitor_cli", log_file=Config.torchLoom_MONITOR_CLI_LOG_FILE)

class MyShell(cmd.Cmd):
    prompt = 'torchLoom> '
    intro = "Welcome to torchLoom CLI"

    def __init__(self, completekey = "tab", stdin = None, stdout = None):
        super().__init__(completekey, stdin, stdout)
        self._loop = asyncio.get_event_loop()
        logger.info("torchLoom CLI initialized")
        self._nc = None

    def do_test(self, line):
        logger.info(f"Executing test command with input: {line}")
        self._loop.run_until_complete(self._send_device_err(line))
    
    def do_setlr(self, line):
        logger.info(f"Executing test command with input: {line}")
        self._loop.run_until_complete(self._send_set_lr(line))
        
    def do_exit(self, line):
        """Exit the CLI"""
        logger.info("Exiting torchLoom CLI")
        if self._nc is not None:
            self._loop.run_until_complete(self._nc.close())
        logger.info("NATS connection closed")
        return True
        
    def do_register_device(self, line):
        """Register a device with a replica
        
        Usage: register_device <device_uuid> <replica_id>
        """
        logger.info(f"Registering device with input: {line}")
        
        parts = line.strip().split()
        if len(parts) != 2:
            logger.error("Invalid format. Usage: register_device <device_uuid> <replica_id>")
            return
            
        device_uuid, replica_id = parts
        
        async def send_register_device(device_uuid, replica_id):
            try:
                if self._nc is None:
                    self._nc = await nats.connect(torchLoomConstants.DEFAULT_ADDR)
                logger.debug(f"Connected to NATS server at {torchLoomConstants.DEFAULT_ADDR}")
                
                register_envelope = EventEnvelope()
                register_envelope.register_device.device_uuid = device_uuid
                register_envelope.register_device.replica_id = replica_id
                
                await self._nc.publish(torchLoomConstants.monitor_stream.subjects.EXTERNAL, register_envelope.SerializeToString())
                logger.info(f"Published device registration for device {device_uuid} with replica {replica_id}")
            except Exception as e:
                logger.exception(f"Failed to register device: {e}")
                
        asyncio.run(send_register_device(device_uuid, replica_id))
        
    def do_fail_device(self, line):
        """Simulate a device failure
        
        Usage: fail_device <device_uuid>
        """
        logger.info(f"Simulating failed GPU with uuid: {line}")
        
        device_uuid = line.strip()
        if not device_uuid:
            logger.error("Device UUID is required")
            return
            
        asyncio.run(self._send_device_err(device_uuid))
        
    def do_reset_lr(self, line):
        """Reset the learning rate
        
        Usage: reset_lr <learning_rate>
        """
        logger.info(f"Resetting learning rate to: {line}")
        
        lr = line.strip()
        if not lr:
            logger.error("Learning rate is required")
            return
            
        asyncio.run(self._send_set_lr(lr))

    async def _send_set_lr(self, line):
        logger.info(f"Simulating reset lr to: {line}")
        try:
            if self._nc is None:
                self._nc = await nats.connect(torchLoomConstants.DEFAULT_ADDR)
            logger.debug(f"Connected to NATS server at {torchLoomConstants.DEFAULT_ADDR}")
            
            # await js.add_stream(name=torchLoomConstants.weaver_stream.STREAM, subjects=[torchLoomConstants.monitor_stream.subjects.EXTERNAL])
            DRenvelope = EventEnvelope()
            DRenvelope.learning_rate.lr = line
            
            await self._nc.publish(torchLoomConstants.monitor_stream.subjects.EXTERNAL, DRenvelope.SerializeToString())
            logger.info(f"Published reset learning rate event with lr equals {line}")

        except Exception as e:
            logger.exception(f"Failed to reset learning rate error: {e}")
    
    async def _send_device_err(self, line):
        logger.info(f"Simulating failed GPU with uuid: {line}")
        try:
            if self._nc is None:
                self._nc = await nats.connect(torchLoomConstants.DEFAULT_ADDR)
            logger.debug(f"Connected to NATS server at {torchLoomConstants.DEFAULT_ADDR}")
            
            # await js.add_stream(name=torchLoomConstants.weaver_stream.STREAM, subjects=[torchLoomConstants.monitor_stream.subjects.EXTERNAL])
            DRenvelope = EventEnvelope()
            DRenvelope.monitored_fail.device_uuid = line
            
            await self._nc.publish(torchLoomConstants.monitor_stream.subjects.EXTERNAL, DRenvelope.SerializeToString())
            logger.info(f"Published device failure event for device {line}")

        except Exception as e:
            logger.exception(f"Failed to send device error: {e}")

    def do_quit(self, line):
        """Exit the CLI and close NATS connection."""
        logger.info("Exiting torchLoom CLI")
        if self._nc is not None:
            self._loop.run_until_complete(self._nc.close())
        logger.info("NATS connection closed")
        return True

    def default(self, line):
        logger.warning(f"Unknown command: {line}")
        
    def emptyline(self):
        """Do nothing on empty line."""
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="torchLoom CLI")
    parser.add_argument("--host", type=str, default="localhost", help="NATS server host")
    parser.add_argument("--port", type=int, default=4222, help="NATS server port")
    args = parser.parse_args()

    if args.host and args.port:
        torchLoomConstants.DEFAULT_ADDR = f"nats://{args.host}:{args.port}"
        logger.info(f"Using NATS server at {torchLoomConstants.DEFAULT_ADDR}")
    
    logger.info("Starting torchLoom CLI")
    MyShell().cmdloop()