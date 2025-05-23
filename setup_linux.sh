curl -LO "https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protoc-25.1-linux-x86_64.zip"
pip install -e .[dev]
pip install -r requirements.txt
go install github.com/nats-io/natscli/nats@latest
curl -sf https://binaries.nats.dev/nats-io/nats-server/v2@v2.10.20 | sh
mv nats-server ./nats/

cd torchLoom-ui
npm install
npm run dev