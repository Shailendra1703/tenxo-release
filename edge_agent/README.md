# Tenxo Edge Agent (Rust)

Runs on GPU provider machines. Connects to the matchmaker, accepts encrypted jobs, executes them in Docker, and returns results.

## Production Setup

### 1. Prerequisites

```bash
# Docker (with NVIDIA support)
sudo apt install docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 2. One-Command Install

```bash
curl -fsSL https://tenxo-api.onrender.com/install.sh | bash -s -- --owner YOUR_USER_ID
```

Replace `YOUR_USER_ID` with your account ID from the Tenxo dashboard.

### 3. Verify

```bash
sudo systemctl status tenxo-agent
sudo journalctl -u tenxo-agent -f
```

Your node appears in the marketplace within 30 seconds.

## Manual Build (no install script)

```bash
git clone https://github.com/Tanya25-05/tenxo.git
cd tenxo/edge_agent
cargo build --release

MATCHMAKER_URL=https://tenxo-api.onrender.com \
OWNER=<your_user_id> \
./target/release/edge_agent
```

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `MATCHMAKER_URL` | `http://127.0.0.1:8080` | Yes | Matchmaker HTTP + WS address |
| `OWNER` | `""` | Yes | Your Supabase user ID from the dashboard |
| `NODE_ID` | `node-<uuid>` | No | Custom node identifier |

## What the Agent Does

1. Detects GPU model + VRAM via `nvidia-smi`
2. Connects to matchmaker signaling WebSocket
3. Performs challenge-response with TEE attestation (or dev mode fallback)
4. Generates ephemeral X25519 keypair for ECDH key exchange
5. Sends heartbeats every 20 seconds
6. Receives jobs → downloads encrypted payload → decrypts → runs in Docker → re-encrypts → uploads result

## Docker Sandbox

Jobs run with maximum isolation:
- `--network none` — no network access
- `--cap-drop ALL` — no Linux capabilities
- `--security-opt no-new-privileges:true`
- Ephemeral temp directory, cleaned up after job
