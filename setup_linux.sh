conda env create -f environment.yaml
conda activate nats-torch27
go install github.com/nats-io/natscli/nats@latest
curl -sf https://binaries.nats.dev/nats-io/nats-server/v2@v2.10.20 | sh
mv nats-server ./nats/