python -m grpc_tools.protoc \
  --proto_path=$(pwd)/torchLoom/proto \
  --python_out=$(pwd)/torchLoom/proto \
  $(pwd)/torchLoom/proto/torchLoom.proto

kill_processes() {
    # Kill processes matching given pattern
    local pattern=$1
    local pids=$(ps aux | grep "$pattern" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        kill -9 $pids
        echo "Killed processes matching pattern: $pattern, pids: $pids"
    fi
}

# Kill existing processes
kill_processes "nats-server -c"

if [ -d "/srv/tmp/jetstream/store" ]; then
    rm -r /srv/tmp/jetstream/store
    echo "Cleared jetstream store"
fi
