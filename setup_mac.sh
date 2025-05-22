conda env create -f environment.yaml
conda activate nats-torch27
brew install nats-io/nats-tools/nats-tools
mv /opt/homebrew/opt/nats-server/bin/nats-server ./nats/
brew install protobuf