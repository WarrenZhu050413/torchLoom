import asyncio
import logging
from nats.aio.client import Client as NATS
from torchLoom.common.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logging.basicConfig(level=logging.DEBUG)

async def test_stream_publish():
    nc = NATS()
    try:
        # Connect to NATS
        await nc.connect('nats://localhost:4222')
        print('✅ Connected to NATS successfully')
        
        js = nc.jetstream()
        
        # Check if the stream exists and its configuration
        try:
            stream_info = await js.stream_info('WEAVELET_STREAM')
            print(f'✅ Found stream: {stream_info.config.name}')
            print(f'✅ Stream subjects: {stream_info.config.subjects}')
        except Exception as e:
            print(f'❌ Stream info failed: {e}')
            return
        
        # Create a test device registration message
        envelope = EventEnvelope()
        envelope.register_device.device_uuid = 'test_device_123'
        envelope.register_device.replica_id = 'test_replica_123'
        
        # Try to publish it
        try:
            await js.publish(
                torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                envelope.SerializeToString(),
            )
            print(f'✅ Successfully published device registration to {torchLoomConstants.weaver_stream.subjects.DR_SUBJECT}')
        except Exception as e:
            print(f'❌ Failed to publish device registration: {e}')
        
        await nc.close()
        return True
    except Exception as e:
        print(f'❌ Failed to connect: {e}')
        return False

if __name__ == "__main__":
    success = asyncio.run(test_stream_publish())
    if success:
        print("Stream publish test completed")
    else:
        print("Stream publish test failed") 