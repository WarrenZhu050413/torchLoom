
<!-- 
4. Set up torchFT by starting the lighthouse

```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

5. Run the torchFT training script on one device

Start Device 1:
```sh
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2

CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 -- train_ddp.py
```

6. Optionally, to test multiple device failures, you can run the training script on another device.

Start Device 2:
```sh
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2

CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 -- train_ddp.py
```

7. Now, control C on any of the training processes. See what happens! Also relaunch the training script, and see what happens!

When the training processes run, you should be able to see the devices registering their device_uuid and process_id.

```sh
Registered device: device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 for process_id: train_ddp_0:d5aa538f-3268-4f78-ae88-3afff894e629 # For replica 0
Registered device: device-307a982d-bf2b-4cc3-64e3-aae456bf6a28 for process_id: train_ddp_1:164ecd9c-f806-4eef-8fd3-add20298ea20 # For replica 1
``` -->