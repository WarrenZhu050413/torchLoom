# Testing

To test, start the monitor cli:
```sh
python /srv/apps/warren/torchft/torchft/torchLoom/monitor_cli.py
torchLoom>
```

## Test 0: DRMapping Registration upon starting training

Weaver output:

```bash
torchLoom constants module loaded
Starting torchLoom Weaver
Weaver initialized with NATS address: nats://0.0.0.0:4222
Connected to NATS server at nats://0.0.0.0:4222
Subscribing to torchLoom.DRentry on stream CONTROLLER-STREAM with consumer weaver-consumer
Subscribed to torchLoom.monitored.failure
Weaver initialized and subscribed to all subjects
Started listening on torchLoom.DRentry
Started listening on torchLoom.monitored.failure
```

Once the two training scrips start according to the above instructions, then should see the registration on the weaver:

```bash
----------------------------------------------------------------------------------------------------
Received register_device event for device device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 and replica train_ddp_1:fe7a317e-6474-4d9d-8c8a-2a74b321af17
New Mapping: Device device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 is now associated with replicas: {'train_ddp_1:fe7a317e-6474-4d9d-8c8a-2a74b321af17'}
New Mapping: Replica train_ddp_1:fe7a317e-6474-4d9d-8c8a-2a74b321af17 is now associated with devices: {'device-307a982d-bf2b-4cc3-64e3-aae456bf6a28'}

----------------------------------------------------------------------------------------------------
Received register_device event for device device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 and replica train_ddp_0:e0d9df8d-8b61-4f2d-a769-5acc59b9ef9d
New Mapping: Device device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 is now associated with replicas: {'train_ddp_0:e0d9df8d-8b61-4f2d-a769-5acc59b9ef9d', 'train_ddp_1:fe7a317e-6474-4d9d-8c8a-2a74b321af17'}
New Mapping: Replica train_ddp_0:e0d9df8d-8b61-4f2d-a769-5acc59b9ef9d is now associated with devices: {'device-307a982d-bf2b-4cc3-64e3-aae456bf6a28'}
```

## Test 1: To fail a replica, run:

```sh
torchLoom> test 50
Received message
[device FAILURE] Device not found in device replica map
```

Here, there are two possible cases. If the device_uuid is not found in the device replica map, then the device is not registered.
Here, the device_uuid is printed out by the weaver whenever a training process starts on a device.

## Test 1a: Device not registered

CLI output:

```bash
torchLoom> test 50
Executing test command with input: 50
Simulating failed device with uuid: 50
Published device failure event for device 50
```

Weaver output:
```bash
[device FAILURE] Device 50 not found in device-to-replicas map
```

## Test 1b: Device registered

CLI output:
```sh
torchLoom> test device-307a982d-bf2b-4cc3-64e3-aae456bf6a28
Executing test command with input: device-307a982d-bf2b-4cc3-64e3-aae456bf6a28
Simulating failed device with uuid: device-307a982d-bf2b-4cc3-64e3-aae456bf6a28
Published device failure event for device device-307a982d-bf2b-4cc3-64e3-aae456bf6a28
```

Weaver output:

```bash
[device FAILURE] Associated Replica IDs: {'train_ddp_1:b584d120-6037-4a33-aeb6-54fcbcbee9bf'}
```

Output from the associated replica: