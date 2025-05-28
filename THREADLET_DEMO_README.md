1. Start the nats server:

```sh
./nats/nats-server -c ./nats/nats.conf
```

2. Start the weaver:

```sh
python -m torchLoom.weaver.weaver
```

3. Start the websocket server:

```sh
python websocket_cli.py
```

4. Start the training process with threadlet

```sh
python spawn_threadlet.py
```