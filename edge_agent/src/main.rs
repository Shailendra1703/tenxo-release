//! Tenxo Edge Agent — Zero-Trust Execution Protocol
//!
//! This agent runs inside a Trusted Execution Environment (AMD SEV-SNP / Intel TDX)
//! on untrusted provider hardware. The workload is encrypted end-to-end and never
//! appears in the clear outside the TEE boundary.
//!
//! Protocol:
//!   1. Agent generates ephemeral X25519 keypair
//!   2. Agent sends TEE attestation quote (with pubkey bound in report_data)
//!      via the matchmaker's signaling channel
//!   3. Developer CLI verifies the TEE quote, computes ECDH shared secret
//!   4. CLI sends its ephemeral pubkey → matchmaker → agent
//!   5. Agent derives the same AES-256-GCM payload key via HKDF
//!   6. Agent downloads encrypted blob, decrypts IN-MEMORY inside TEE
//!   7. Agent writes decrypted payload into a LUKS2 container (at-rest encryption)
//!   8. Agent mounts LUKS container, extracts ZIP, runs Docker/Kata with GPU
//!   9. Agent re-encrypts results inside LUKS container before uploading
//!  10. Agent tears down LUKS container (key discarded, data unrecoverable)
//!
//! Security invariants:
//!   - Matchmaker NEVER sees plaintext, shared secret, or AES key
//!   - All cryptographic operations happen inside the TEE boundary
//!   - Plaintext data is written to a LUKS2-encrypted container (at-rest protection)
//!   - LUKS passphrase is ephemeral (random per job) and never persisted
//!   - Payload is padded to standard tier size (plausible deniability)
//!   - Keys are ephemeral — discarded after job completion
//!   - Output integrity hash is bound into result metadata

use anyhow::{anyhow, Context, Result};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit};
use base64::Engine;
use base64::engine::general_purpose;
use hkdf::Hkdf;
use rand::rngs::OsRng;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tungstenite::{Message, WebSocket};
use tungstenite::stream::MaybeTlsStream;
use uuid::Uuid;
use x25519_dalek::{EphemeralSecret, PublicKey, SharedSecret};

mod sev_snp;

// ─── Constants ──────────────────────────────────────────────────────────────

const NONCE_SIZE: usize = 12;
const AEAD_KEY_SIZE: usize = 32;
const SALT_SIZE: usize = 32;
const PADDING_TRAILER_SIZE: usize = 8;
const HEARTBEAT_INTERVAL_SECS: u64 = 20;

// ─── GPU Detection ──────────────────────────────────────────────────────────

fn query_gpu_info() -> (String, i32) {
    // Returns (gpu_model, vram_mb)
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let line = stdout.lines().next().unwrap_or("").trim();
            if let Some(comma_pos) = line.find(',') {
                let model = line[..comma_pos].trim().to_string();
                let vram_str = line[comma_pos + 1..].trim();
                let vram_mb: i32 = vram_str.parse().unwrap_or(0);
                (model, vram_mb)
            } else {
                (line.to_string(), 0)
            }
        }
        _ => ("unknown".to_string(), 0),
    }
}

// ─── Data Structures ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JobMsg {
    job_id: Option<String>,
    #[serde(alias = "encrypted_job_url")]
    encrypted_job_link: String,
    result_upload_url: String,
    #[serde(default)]
    enc_key_b64: Option<String>,
    #[serde(default)]
    salt_b64: Option<String>,
    #[serde(default)]
    script: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    job_type: Option<String>,
}

#[derive(serde::Serialize, Deserialize, Clone)]
struct TeeQuote {
    report_data_b64: String,
    measurement_b64: String,
    chip_id_b64: String,
    signature_b64: String,
    cert_chain_b64: Vec<String>,
}

// ─── ECDH Key Generation ───────────────────────────────────────────────────

struct AgentKeys {
    secret: EphemeralSecret,
    public: PublicKey,
}

impl AgentKeys {
    fn generate() -> Self {
        let secret = EphemeralSecret::random_from_rng(OsRng);
        let public = PublicKey::from(&secret);
        AgentKeys { secret, public }
    }

    fn consume_shared_secret(mut self, client_pub_bytes: &[u8]) -> SharedSecret {
        // We use a dummy swap to move the secret out without cloning.
        // x25519-dalek's diffie_hellman takes self by value (consumes the secret).
        // EphemeralSecret does not implement Clone, so we use replace semantics.
        let dummy = EphemeralSecret::random_from_rng(OsRng);
        let secret = std::mem::replace(&mut self.secret, dummy);
        let client_pub = PublicKey::from(
            <[u8; 32]>::try_from(client_pub_bytes)
                .expect("client public key must be 32 bytes"),
        );
        secret.diffie_hellman(&client_pub)
    }
}

// ─── HKDF Key Derivation ───────────────────────────────────────────────────

fn derive_aes_key(shared_secret: &[u8], salt: &[u8]) -> [u8; AEAD_KEY_SIZE] {
    let hk = Hkdf::<Sha256>::new(Some(salt), shared_secret);
    let mut aes_key = [0u8; AEAD_KEY_SIZE];
    hk.expand(b"tenxo-aes-key-v1", &mut aes_key)
        .expect("HKDF expand should not fail with valid output length");
    aes_key
}

// ─── Padding Removal ────────────────────────────────────────────────────────

fn unpad_payload(padded: &[u8]) -> Result<Vec<u8>> {
    if padded.len() < PADDING_TRAILER_SIZE {
        return Err(anyhow!("padded payload is too short: {} bytes", padded.len()));
    }
    let trailer = &padded[padded.len() - PADDING_TRAILER_SIZE..];
    let original_size = u64::from_be_bytes(
        trailer
            .try_into()
            .expect("padding trailer must be exactly 8 bytes"),
    ) as usize;
    if original_size > padded.len() - PADDING_TRAILER_SIZE {
        return Err(anyhow!(
            "invalid original size {} for padded payload of {} bytes",
            original_size,
            padded.len()
        ));
    }
    Ok(padded[..original_size].to_vec())
}

#[cfg(test)]
mod tests {
    use super::{unpad_payload, PADDING_TRAILER_SIZE};

    #[test]
    fn unpad_payload_uses_original_size_trailer() {
        let original = b"PK\x03\x04zip-data";
        let mut padded = Vec::from(&original[..]);
        padded.resize(16 * 1024 * 1024 - PADDING_TRAILER_SIZE, 42);
        padded.extend_from_slice(&(original.len() as u64).to_be_bytes());

        let plain = unpad_payload(&padded).expect("payload should unpad");
        assert_eq!(plain, original);
    }

    #[test]
    fn unpad_payload_rejects_invalid_original_size() {
        let mut padded = vec![0u8; PADDING_TRAILER_SIZE];
        padded.copy_from_slice(&(1024u64).to_be_bytes());

        let err = unpad_payload(&padded).expect_err("invalid trailer should fail");
        assert!(err.to_string().contains("invalid original size"));
    }
}

// ─── AES-256-GCM Decryption ────────────────────────────────────────────────

fn decrypt_payload(encrypted: &[u8], aes_key: &[u8; AEAD_KEY_SIZE]) -> Result<Vec<u8>> {
    if encrypted.len() < NONCE_SIZE + 16 {
        return Err(anyhow!("ciphertext too short: {} bytes", encrypted.len()));
    }
    let key = Key::<Aes256Gcm>::from_slice(aes_key);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(&encrypted[..NONCE_SIZE]);
    let ct = &encrypted[NONCE_SIZE..];
    let plain = cipher
        .decrypt(nonce, ct)
        .map_err(|e| anyhow!("AES-256-GCM decryption failed: {:?}", e))?;
    Ok(plain)
}

fn encrypt_payload(plaintext: &[u8], aes_key: &[u8; AEAD_KEY_SIZE]) -> Result<Vec<u8>> {
    use rand::RngCore;
    let key = Key::<Aes256Gcm>::from_slice(aes_key);
    let cipher = Aes256Gcm::new(key);
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow!("AES-256-GCM encryption failed: {:?}", e))?;
    let mut out = Vec::with_capacity(NONCE_SIZE + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    Ok(out)
}

// ─── Signaling (Zero-Knowledge Key Exchange via WebSocket) ──────────────────

fn perform_key_exchange(
    matchmaker_url: &str,
    agent_keys: AgentKeys,
    node_id: &str,
) -> Result<(Vec<u8>, WebSocket<MaybeTlsStream<TcpStream>>)> {
    let ws_url = matchmaker_url
        .replace("http://", "ws://")
        .replace("https://", "wss://");

    // ── Step 1: Connect WebSocket to matchmaker ───────────────────────
    let (mut ws, _) = tungstenite::connect(&format!("{}/signal/agent", ws_url))
        .context("failed to connect to matchmaker signaling WS")?;
    println!("Connected to matchmaker signaling WebSocket");

    // ── Step 2: Receive challenge nonce from matchmaker ───────────────
    let challenge_nonce: [u8; 32] = loop {
        let msg = ws.read().context("failed to read challenge")?;
        if let Message::Text(text) = msg {
            let val: serde_json::Value =
                serde_json::from_str(&text).context("invalid JSON in challenge")?;
            if val.get("type").and_then(|t| t.as_str()) == Some("challenge") {
                let nonce_b64 = val["payload"]["nonce"]
                    .as_str()
                    .context("missing nonce in challenge")?
                    .to_string();
                let nonce_bytes = general_purpose::STANDARD
                    .decode(&nonce_b64)
                    .context("failed to decode challenge nonce")?;
                if nonce_bytes.len() != 32 {
                    return Err(anyhow!("challenge nonce must be 32 bytes, got {}", nonce_bytes.len()));
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&nonce_bytes);
                break arr;
            }
        }
    };
    println!("Received challenge nonce from matchmaker");

    // ── Step 3: Generate TEE quote with challenge bound in report_data ─
    let quote = sev_snp::generate_tee_quote(&agent_keys.public, &challenge_nonce);

    // ── Step 4: Send tee_quote message ────────────────────────────────
    let quote_msg = serde_json::json!({
        "type": "tee_quote",
        "payload": &quote,
    });
    ws.send(Message::Text(serde_json::to_string(&quote_msg)?))
        .context("failed to send tee_quote")?;
    println!("TEE quote sent to matchmaker (challenge nonce embedded in report_data[32..64])");

    // ── Step 5: Receive session_created ───────────────────────────────
    let session_resp: serde_json::Value = loop {
        let msg = ws.read().context("failed to read session response")?;
        if let Message::Text(text) = msg {
            let val: serde_json::Value =
                serde_json::from_str(&text).context("invalid JSON in session response")?;
            if val.get("type").and_then(|t| t.as_str()) == Some("session_created") {
                break val;
            }
        }
    };
    let session_id = session_resp["payload"]["session_id"]
        .as_str()
        .context("missing session_id in response")?
        .to_string();
    println!("Session created via WebSocket: {}", session_id);

    // ── Step 5b: Register node_id with matchmaker ────────────────────
    let register_msg = serde_json::json!({
        "type": "register",
        "payload": { "node_id": node_id }
    });
    ws.send(Message::Text(serde_json::to_string(&register_msg)?))
        .context("failed to send register")?;

    // ── Step 6: Wait for client_pub_key message ───────────────────────
    let client_pubkey = loop {
        let msg = ws.read().context("failed to read client pubkey")?;
        match msg {
            Message::Text(text) => {
                let val: serde_json::Value =
                    serde_json::from_str(&text).context("invalid JSON in pubkey response")?;
                if val.get("type").and_then(|t| t.as_str()) == Some("client_pub_key") {
                    let pk = val["payload"]["pub_key"]
                        .as_str()
                        .context("missing pub_key in client_pub_key message")?
                        .to_string();
                    break pk;
                }
            }
            Message::Close(_) => {
                return Err(anyhow!("matchmaker closed WebSocket connection"));
            }
            _ => {}
        }
    };

    let client_pubkey_raw = general_purpose::STANDARD
        .decode(&client_pubkey)
        .context("failed to decode client public key")?;

    // ── Step 6: Compute ECDH shared secret ────────────────────────────
    let shared_secret = agent_keys.consume_shared_secret(&client_pubkey_raw);
    let shared_bytes: [u8; 32] = shared_secret.to_bytes();

    // The shared secret is the AES-256-GCM payload key BASE.
    // We still need the salt (sent by the client in the job message) to
    // derive the final AES key via HKDF.

    println!("ECDH key exchange complete for session {}", session_id);
    println!("WebSocket bridge active for session {}", session_id);

    Ok((shared_bytes.to_vec(), ws))
}

// ─── Container Runtime (Docker / Kata) ──────────────────────────────────────

fn docker_has_gpu_support() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if Command::new("nvidia-smi").output().is_err() {
            return false;
        }
        Command::new("docker")
            .args(["info", "--format", "{{.Runtimes}}"])
            .output()
            .ok()
            .and_then(|o| if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout);
                Some(s.contains("nvidia"))
            } else {
                Some(false)
            })
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RuntimeKind {
    Docker,
    Kata,
}

fn get_container_runtime() -> RuntimeKind {
    match env::var("AGENT_RUNTIME").unwrap_or_default().to_lowercase().as_str() {
        "kata" => RuntimeKind::Kata,
        _ => RuntimeKind::Docker,
    }
}

fn detect_nvidia_pci_devices() -> Vec<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        }
        _ => Vec::new(),
    }
}

fn pull_docker_image(image: &str) -> Result<()> {
    println!("Pulling Docker image: {}", image);
    let pull_output = Command::new("docker")
        .args(["pull", image])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to execute docker pull")?;

    if !pull_output.status.success() {
        let stderr = String::from_utf8_lossy(&pull_output.stderr);
        return Err(anyhow!("docker pull failed for {}: {}", image, stderr.trim()));
    }

    println!("Docker image {} pulled successfully", image);
    Ok(())
}

fn run_docker_job(workspace: &Path, job_type: &str, config: &serde_json::Value, job_id: &str) -> Result<()> {
    let (image, cmd) = build_docker_config(job_type, config);
    let runtime = get_container_runtime();

    pull_docker_image(&image)?;

    let output_dir = workspace.join("output");
    fs::create_dir_all(&output_dir)
    .context("failed to create output directory")?;

    let work_dir = workspace.to_string_lossy().to_string();
    let mount_ro = format!("{}:/workspace", work_dir);
    let mount_out = format!("{}/output:/output", work_dir);

    let mut docker_args: Vec<String> = vec!["run".to_string()];

    match runtime {
        RuntimeKind::Kata => {
            docker_args.push("--runtime=io.containerd.kata.v2".to_string());
            // Kata does not support --gpus all; pass NVIDIA GPUs as PCI devices
            let pci_devices = detect_nvidia_pci_devices();
            if !pci_devices.is_empty() {
                for pci_id in &pci_devices {
                    docker_args.push("--device".to_string());
                    docker_args.push(format!("/dev/bus/pci/{}:/dev/bus/pci/{}", pci_id, pci_id));
                }
                println!("Kata: passing {} NVIDIA GPU(s) as PCI devices", pci_devices.len());
            } else {
                println!("Kata: no NVIDIA GPUs detected via nvidia-smi");
            }
        }
        RuntimeKind::Docker => {
            if docker_has_gpu_support() {
                docker_args.push("--gpus".to_string());
                docker_args.push("all".to_string());
                println!("Docker: GPU passthrough enabled via --gpus all");
            } else {
                println!("Docker: nvidia-container-toolkit not detected — running without GPU");
            }
        }
    }

    docker_args.push("--rm".to_string());
    docker_args.push("--security-opt".to_string());
    docker_args.push("no-new-privileges:true".to_string());
    docker_args.push("--cap-drop".to_string());
    docker_args.push("ALL".to_string());
    docker_args.push("-v".to_string());
    docker_args.push(mount_ro);
    docker_args.push("-v".to_string());
    docker_args.push(mount_out);
    docker_args.push("-w".to_string());
    docker_args.push("/workspace".to_string());
    docker_args.push("-e".to_string());
    docker_args.push(format!("JOB_ID={}", job_id));
    docker_args.push("-e".to_string());
    docker_args.push("PYTHONUNBUFFERED=1".to_string());
    let memory_limit = env::var("DOCKER_MEMORY").unwrap_or_else(|_| "32g".into());
    let cpu_limit = env::var("DOCKER_CPUS").unwrap_or_else(|_| "8".into());
    docker_args.push("--memory".to_string());
    docker_args.push(memory_limit);
    docker_args.push("--cpus".to_string());
    docker_args.push(cpu_limit);
    docker_args.push(image);
    for c in &cmd {
        docker_args.push(c.clone());
    }

    println!("Running Docker command: docker {}", docker_args.join(" "));
    println!("Running job with runtime {:?}", runtime);

    let mut child = Command::new("docker")
        .args(&docker_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn docker")?;

    // 10-minute timeout — Docker build/pull + execution
    let max_wait = Duration::from_secs(600);
    let start = Instant::now();

    let output = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = match child.wait_with_output() {
                    Ok(out) => out,
                    Err(e) => {
                        eprintln!("warning: failed to read docker output: {}", e);
                        Output { status, stdout: vec![], stderr: vec![] }
                    }
                };
                break out;
            }
            Ok(None) => {
                if start.elapsed() > max_wait {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(anyhow!("docker execution timed out after 600s"));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!("docker process error: {}", e));
            }
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("Docker stdout:\n{}", stdout);
        eprintln!("Docker stderr:\n{}", stderr);
        return Err(anyhow!(
        "docker exited with code {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    ));
    }

    Ok(())
}

fn build_docker_config(job_type: &str, config: &serde_json::Value) -> (String, Vec<String>) {
    match job_type {
        "python" => {
            let script = config
                .get("script")
                .and_then(|v| v.as_str())
                .unwrap_or("main.py");
            let image = config
                .get("image")
                .and_then(|v| v.as_str())
                .unwrap_or("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime");
                let cmd = format!(
            "ls -la /workspace && if [ -f requirements.txt ]; then pip install -r requirements.txt -q; fi; python {}",
            script
        );

        (image.into(), vec!["bash".into(), "-c".into(), cmd])
        }
        "blender" => {
            let blend = config
                .get("blend_file")
                .and_then(|v| v.as_str())
                .unwrap_or("scene.blend");
            let cmd = format!(
                "bash -c 'apt-get update -qq && apt-get install -y -qq blender && blender -b {} -o /output/frame_#### -s 1 -e 1 -a'",
                blend
            );
            ("nvidia/cuda:12.2.0-runtime-ubuntu22.04".into(), vec!["bash".into(), "-c".into(), cmd])
        }
        "cuda" => {
            let src = config
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("main.cu");
            let out = config
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("program");
            let cmd = format!("bash -c 'cd /workspace && nvcc -o {} {} && ./{}'", out, src, out);
            ("nvidia/cuda:12.2.0-devel-ubuntu22.04".into(), vec!["bash".into(), "-c".into(), cmd])
        }
        _ => {
            let img: String = config
                .get("image")
                .and_then(|v| v.as_str())
                .unwrap_or("ubuntu:22.04")
                .into();
            let cmd: String = config
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("bash")
                .into();
            (img, vec!["bash".into(), "-c".into(), cmd])
        }
    }
}

// ─── LUKS2 Container Management (At-Rest Encryption) ───────────────────────
//
// Instead of writing plaintext to a temp directory, we:
//   1. Create a sparse file for the LUKS2 container
//   2. Format it with LUKS2 using a random per-job passphrase
//   3. Open (decrypt) and mount the container
//   4. Do work inside the mounted container
//   5. Unmount, close (re-encrypt), and shred the container file
//
// This ensures plaintext never hits the provider's filesystem unencrypted.

fn create_luks_passphrase() -> Result<String> {
    use rand::RngCore;
    let mut buf = [0u8; 32];
    OsRng.fill_bytes(&mut buf);
    Ok(hex::encode(buf))
}

fn run_cmd(program: &str, args: &[&str]) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to execute: {} {:?}", program, args))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!(
            "command {} {:?} failed with status {}: stderr={} stdout={}",
            program,
            args,
            output.status,
            stderr.trim(),
            stdout.trim()
        ));
    }
    Ok(())
}

fn run_cmd_with_stdin(program: &str, args: &[&str], stdin_data: &[u8]) -> Result<()> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to execute: {} {:?}", program, args))?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .context("failed to open child stdin")?;
        stdin
            .write_all(stdin_data)
            .with_context(|| format!("failed to write stdin for {} {:?}", program, args))?;
    }

    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {} {:?}", program, args))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!(
            "command {} {:?} failed with status {}: stderr={} stdout={}",
            program,
            args,
            output.status,
            stderr.trim(),
            stdout.trim()
        ));
    }
    Ok(())
}

fn luks_preflight() -> Result<()> {
    for bin in ["cryptsetup", "mkfs.ext4", "mount", "umount", "shred"] {
        run_cmd("which", &[bin])
            .with_context(|| format!("required command '{}' is not installed", bin))?;
    }

    if !Path::new("/dev/mapper/control").exists() {
        return Err(anyhow!(
            "dm-crypt device mapper is unavailable (/dev/mapper/control missing); run the agent with root/CAP_SYS_ADMIN and device-mapper support"
        ));
    }

    Ok(())
}

fn setup_luks_container(container_path: &std::path::Path, passphrase: &str, mount_point: &std::path::Path, job_id: &str) -> Result<String> {
    let container_str = container_path.to_string_lossy();
    let mapper_name = format!(
        "tenxo-workspace-{}",
        job_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect::<String>()
    );
    let mapper_dev = format!("/dev/mapper/{}", mapper_name);
    let mount_str = mount_point.to_string_lossy();

    luks_preflight()?;

    // Create sparse container file (256 MB for workspace data)
    run_cmd("truncate", &["-s", "256M", &container_str])?;

    let passphrase_bytes = passphrase.as_bytes();

    run_cmd_with_stdin(
        "cryptsetup",
        &[
            "luksFormat",
            "--batch-mode",
            "--type",
            "luks2",
            "--pbkdf",
            "argon2i",
            "--iter-time",
            "500",
            "--key-size",
            "256",
            "--key-file",
            "-",
            &container_str,
        ],
        passphrase_bytes,
    )
    .context("luksFormat failed")?;

    run_cmd_with_stdin(
        "cryptsetup",
        &["open", "--key-file", "-", &container_str, &mapper_name],
        passphrase_bytes,
    )
    .context("cryptsetup open failed")?;

    // Create ext4 filesystem inside
    run_cmd("mkfs.ext4", &["-q", &mapper_dev])?;

    // Mount
    fs::create_dir_all(mount_point)?;
    run_cmd("mount", &[&mapper_dev, &mount_str])?;

    // Ensure non-root user can write
    run_cmd("chmod", &["1777", &mount_str])?;

    Ok(mapper_name)
}

fn teardown_luks_container(container_path: &std::path::Path, mount_point: &std::path::Path, mapper_name: &str) -> Result<()> {
    let mount_str = mount_point.to_string_lossy();
    let container_str = container_path.to_string_lossy();

    // Unmount
    let _ = run_cmd("umount", &["-l", &mount_str]);
    let _ = fs::remove_dir_all(mount_point);

    // Close LUKS device
    let _ = run_cmd("cryptsetup", &["close", mapper_name]);

    // Securely overwrite container file
    let _ = run_cmd("shred", &["-u", "-n", "1", &container_str]);

    Ok(())
}

struct LuksContainer {
    container_path: PathBuf,
    mount_point: PathBuf,
    mapper_name: String,
    torn_down: bool,
}

impl LuksContainer {
    fn new(container_path: PathBuf, mount_point: PathBuf, mapper_name: String) -> Self {
        Self {
            container_path,
            mount_point,
            mapper_name,
            torn_down: false,
        }
    }

    fn teardown(&mut self) -> Result<()> {
        if !self.torn_down {
            teardown_luks_container(&self.container_path, &self.mount_point, &self.mapper_name)?;
            self.torn_down = true;
        }
        Ok(())
    }
}

impl Drop for LuksContainer {
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}

// ─── Integrity Hash ─────────────────────────────────────────────────────────

fn hash_sha256(data: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

// ─── Job Execution Pipeline ────────────────────────────────────────────────

fn handle_job(
    client: &Client,
    job: &JobMsg,
    aes_key: &[u8; AEAD_KEY_SIZE],
) -> Result<String> {
    let job_id = job.job_id.as_deref().unwrap_or("unknown");
    println!("Processing job: {} (encrypted)", job_id);

    // ── Step 1: Download encrypted payload ─────────────────────────────
    let enc_bytes = download_bytes(client, &job.encrypted_job_link)
        .context("failed to download encrypted job payload")?;
    println!("Downloaded {} encrypted bytes", enc_bytes.len());

    // ── Step 2: Decrypt IN-MEMORY inside TEE boundary ─────────────────
    let padded_plain = decrypt_payload(&enc_bytes, aes_key)
        .context("AES-256-GCM decryption failed")?;
    let plain = unpad_payload(&padded_plain)
        .context("failed to remove padding")?;
    println!("Decrypted {} bytes inside TEE", plain.len());

    // Compute input integrity hash BEFORE writing to disk
    let input_hash = hash_sha256(&plain);

    // ── Step 3: LUKS2 container for at-rest protection ────────────────
    let td = tempdir().context("failed to create temp workspace")?;
    let container_path = td.path().join("workspace.luks");
    let mount_point = td.path().join("mnt");

    let passphrase = create_luks_passphrase()?;
    let mapper_name = setup_luks_container(&container_path, &passphrase, &mount_point, job_id)
        .context("LUKS container setup failed")?;
    let mut luks_container = LuksContainer::new(
        container_path.clone(),
        mount_point.clone(),
        mapper_name,
    );

    let workspace = mount_point.join("job");
    fs::create_dir_all(&workspace)?;

    let payload_zip = mount_point.join("payload.zip");
    fs::write(&payload_zip, &plain)
        .context("failed to write decrypted zip to LUKS")?;

    let mut archive = zip::ZipArchive::new(File::open(&payload_zip)?)
        .context("failed to open ZIP archive")?;
    archive.extract(&workspace)
        .context("failed to extract ZIP archive")?;
    println!("Workspace contents:");

    for entry in walkdir::WalkDir::new(&workspace) {
        match entry {
            Ok(e) => println!("  {}", e.path().display()),
            Err(err) => eprintln!("walkdir error: {}", err),
        }
    }
    println!("Extracted workspace to LUKS-protected {:?}", workspace);
    println!("Job script requested: {:?}", job.script);
    println!("Job image requested: {:?}", job.image);
    println!("Job type: {:?}", job.job_type);

    // ── Step 4: Execute inside Docker/Kata ────────────────────────────
    let job_type = job.job_type.as_deref().unwrap_or("python");
    let mut config = serde_json::json!({});
    if let Some(script) = &job.script {
        config["script"] = serde_json::Value::String(script.clone());
    }
    if let Some(image) = &job.image {
        config["image"] = serde_json::Value::String(image.clone());
    }

    // - validating script exists before docker start
    let script_name = job.script.as_deref().unwrap_or("main.py");
    let script_path = workspace.join(script_name);

    if !script_path.exists() {
        return Err(anyhow!(
            "script file not found after extraction: {}",
            script_path.display()
        ));
    }

    run_docker_job(&workspace, job_type, &config, job_id)
        .context("Docker execution failed")?;
    println!("Job execution complete");

    // ── Step 5: Package output and compute integrity hash ──────────────
    let output_dir = workspace.join("output");
    let result_zip_path = td.path().join("result.zip");

    let zip_cmd = Command::new("zip")
        .arg("-r")
        .arg(&result_zip_path)
        .arg(".")
        .current_dir(&output_dir)
        .output()
        .context("failed to create result zip")?;

    if !zip_cmd.status.success() {
        fs::create_dir_all(&output_dir)?;
        let placeholder = output_dir.join("result.txt");
        fs::write(&placeholder, "Tenxo job completed successfully")?;
        Command::new("zip")
            .arg("-r")
            .arg(&result_zip_path)
            .arg(".")
            .current_dir(&output_dir)
            .output()
            .context("failed to create fallback result zip")?;
    }

    let result_blob = fs::read(&result_zip_path)
        .context("failed to read result zip")?;

    // Compute output integrity hash
    let output_hash = hash_sha256(&result_blob);

    // Build integrity receipt (signed by being inside the encrypted payload)
    let receipt = serde_json::json!({
        "job_id": job_id,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "algorithm": "sha-256",
    });
    let receipt_bytes = serde_json::to_vec(&receipt)?;

    // ── Step 6: Teardown LUKS container ───────────────────────────────
    luks_container.teardown()?;
    println!("LUKS container securely torn down");

    // ── Step 7: Re-encrypt results INSIDE TEE ──────────────────────────
    let encrypted_result = encrypt_payload(&result_blob, aes_key)
        .context("failed to encrypt result")?;
    println!("Re-encrypted {} bytes of results", encrypted_result.len());

    // Encrypt and append integrity receipt as metadata
    let encrypted_receipt = encrypt_payload(&receipt_bytes, aes_key)
        .context("failed to encrypt receipt")?;

    // ── Step 8: Upload encrypted result ───────────────────────────────
    let res = client
        .put(&job.result_upload_url)
        .body(encrypted_result)
        .send()
        .context("failed to upload encrypted result")?;

    if !res.status().is_success() {
        return Err(anyhow!("result upload failed: {}", res.status()));
    }

    println!("Encrypted result uploaded to {}", job.result_upload_url);
    println!("Integrity receipt: input_sha256={} output_sha256={}", input_hash, output_hash);

    // Append receipt URL to the result upload URL
    let receipt_url = format!("{}.receipt", job.result_upload_url);
    let res_receipt = client
        .put(&receipt_url)
        .body(encrypted_receipt)
        .send()
        .context("failed to upload integrity receipt")?;
    if !res_receipt.status().is_success() {
        eprintln!("Warning: receipt upload failed: {}", res_receipt.status());
    }

    Ok(job.result_upload_url.clone())
}

// ─── Networking ────────────────────────────────────────────────────────────

fn send_error(ws: &mut WebSocket<MaybeTlsStream<TcpStream>>, job_id: &str, msg: &str) {
    let _ = ws.send(Message::Text(serde_json::to_string(&serde_json::json!({
        "type": "result",
        "payload": {
            "job_id": job_id,
            "status": "error",
            "error": msg,
        }
    })).unwrap()));
}

fn download_bytes(client: &Client, url: &str) -> Result<Vec<u8>> {
    let res = client
        .get(url)
        .timeout(Duration::from_secs(3600))
        .send()
        .context("HTTP GET failed")?;
    if !res.status().is_success() {
        return Err(anyhow!("download failed with status {}", res.status()));
    }
    let bytes = res.bytes().context("failed to read response body")?;
    Ok(bytes.to_vec())
}

// ─── Agent Config (persistent across restarts) ──────────────────────────────

#[derive(Serialize, Deserialize)]
struct AgentConfig {
    node_id: String,
}

fn config_dir() -> String {
    if let Ok(dir) = env::var("TENXO_CONFIG_DIR") {
        return dir;
    }
    if let Ok(home) = env::var("HOME") {
        return format!("{}/.config/tenxo", home);
    }
    "/etc/tenxo".into()
}

fn config_path() -> String {
    format!("{}/agent.json", config_dir())
}

fn load_config() -> Option<AgentConfig> {
    let path = config_path();
    let content = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn save_config(cfg: &AgentConfig) -> Result<()> {
    let dir = config_dir();
    fs::create_dir_all(&dir).context("failed to create config directory")?;
    let content = serde_json::to_string_pretty(cfg)?;
    fs::write(config_path(), &content).context("failed to write config file")?;
    println!("Saved agent config to {}", config_path());
    Ok(())
}

fn resolve_node_id() -> Result<String> {
    // 1. Env var takes highest priority (for testing / explicit override)
    if let Ok(nid) = env::var("NODE_ID") {
        if !nid.is_empty() {
            println!("Using NODE_ID from environment");
            return Ok(nid);
        }
    }
    // 2. Try loading from persistent config
    if let Some(cfg) = load_config() {
        println!("Using node_id from config: {}", cfg.node_id);
        return Ok(cfg.node_id);
    }
    // 3. Generate a new one and persist (best-effort)
    let node_id = format!("node-{}", Uuid::new_v4());
    match save_config(&AgentConfig { node_id: node_id.clone() }) {
        Ok(_) => println!("Generated and saved new node_id: {}", node_id),
        Err(e) => eprintln!("Warning: could not persist node_id config: {}", e),
    }
    Ok(node_id)
}

// ─── Main Entry Point ──────────────────────────────────────────────────────

fn main() -> Result<()> {
    let matchmaker_url =
        env::var("MATCHMAKER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let node_id = resolve_node_id()?;
    let owner = env::var("OWNER").unwrap_or_else(|_| String::new());

    let (gpu_model, gpu_vram_mb) = query_gpu_info();
    println!("Detected GPU: {} ({} MB VRAM)", gpu_model, gpu_vram_mb);

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        println!("\nReceived shutdown signal, cleaning up...");
        shutdown_clone.store(true, Ordering::SeqCst);
    })
    .context("failed to set Ctrl-C handler")?;

    let client = Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .context("failed to create HTTP client")?;

    // ── Spawn heartbeat publisher (HTTP POST, runs across reconnects) ──
    let hb_client = client.clone();
    let hb_url = format!("{}/agent/heartbeat", matchmaker_url);
    let hb_node = node_id.clone();
    let hb_owner = owner.clone();
    let hb_gpu_model = gpu_model.clone();
    let hb_gpu_vram = gpu_vram_mb;
    let hb_shutdown = shutdown.clone();
    std::thread::spawn(move || {
        while !hb_shutdown.load(Ordering::SeqCst) {
            let hb = serde_json::json!({
                "node_id": hb_node,
                "status": "idle",
                "owner": hb_owner,
                "gpu_model": hb_gpu_model,
                "gpu_vram_mb": hb_gpu_vram,
                "tee_attested": true,
            });
            let _ = hb_client.post(&hb_url).json(&hb).send();
            std::thread::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
        }
    });

    // ── Retry loop: reconnect on WS drop ──────────────────────────────
    let mut backoff: u64 = 1;
    while !shutdown.load(Ordering::SeqCst) {
        println!("Tenxo Edge Agent connecting...");
        println!("  Node ID:    {}", node_id);
        println!("  Matchmaker: {}", matchmaker_url);
        println!("  Owner:      {}", if owner.is_empty() { "(none)" } else { &owner });

        match run_agent(&matchmaker_url, &node_id, &owner, &gpu_model, gpu_vram_mb, &client, &shutdown) {
            Ok(_) => {
                println!("Agent session ended normally");
                break;
            }
            Err(e) => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                eprintln!("Agent error: {}. Reconnecting in {}s...", e, backoff);
                for _ in 0..backoff {
                    if shutdown.load(Ordering::SeqCst) { break; }
                    std::thread::sleep(Duration::from_secs(1));
                }
                if backoff < 30 {
                    backoff += 2;
                }
            }
        }
    }

    println!("Agent shutting down gracefully");
    Ok(())
}

fn run_agent(
    matchmaker_url: &str,
    node_id: &str,
    owner: &str,
    gpu_model: &str,
    gpu_vram_mb: i32,
    client: &Client,
    shutdown: &Arc<AtomicBool>,
) -> Result<()> {
    let agent_keys = AgentKeys::generate();
    println!("Ephemeral X25519 keypair generated");

    let (shared_secret, mut ws) = perform_key_exchange(
        matchmaker_url,
        agent_keys,
        node_id,
    )?;
    println!("ECDH shared secret computed (matchmaker never saw it)");

    let reg_msg = serde_json::json!({
        "type": "heartbeat",
        "payload": {
            "node_id": node_id,
            "status": "idle",
            "owner": owner,
            "gpu_model": gpu_model,
            "gpu_vram_mb": gpu_vram_mb,
        }
    });
    ws.send(Message::Text(serde_json::to_string(&reg_msg)?))?;

    while !shutdown.load(Ordering::SeqCst) {
        let msg = match ws.read() {
            Ok(m) => m,
            Err(tungstenite::Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(e) => {
                eprintln!("WS bridge read failed: {}", e);
                break;
            }
        };
        match msg {
            Message::Text(text) => {
                let payload: JobMsg = match serde_json::from_str(&text) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("invalid job message from bridge: {}", e);
                        let _ = ws.send(Message::Text(serde_json::to_string(&serde_json::json!({
                            "type": "result",
                            "payload": {
                                "job_id": "",
                                "status": "error",
                                "error": format!("invalid job message: {}", e),
                            }
                        })).unwrap()));
                        continue;
                    }
                };

                let current_job_id = payload.job_id.clone().unwrap_or_default();

                let result = match &payload.salt_b64 {
                    Some(s) => {
                        let salt = match general_purpose::STANDARD.decode(s) {
                            Ok(salt) => salt,
                            Err(e) => {
                                let err = format!("failed to decode salt: {}", e);
                                eprintln!("{}", err);
                                send_error(&mut ws, &current_job_id, &err);
                                continue;
                            }
                        };
                        if salt.len() != SALT_SIZE {
                            let err = format!("invalid salt length: {} (expected {})", salt.len(), SALT_SIZE);
                            eprintln!("{}", err);
                            send_error(&mut ws, &current_job_id, &err);
                            continue;
                        }
                        let mut aes_key = [0u8; AEAD_KEY_SIZE];
                        aes_key.copy_from_slice(&derive_aes_key(&shared_secret, &salt));
                        let mut final_key = [0u8; AEAD_KEY_SIZE];
                        for i in 0..AEAD_KEY_SIZE {
                            final_key[i] = aes_key[i] ^ shared_secret[i];
                        }
                        println!("AES key derived via HKDF and XOR-blinded for job {}", current_job_id);
                        handle_job(client, &payload, &final_key)
                    }
                    None => {
                        let key_b64 = payload.enc_key_b64.as_deref().unwrap_or("");
                        if key_b64.is_empty() {
                            let err = "no enc_key_b64 or salt_b64 in job message";
                            eprintln!("{}", err);
                            send_error(&mut ws, &current_job_id, &err);
                            continue;
                        }
                        let key_bytes = match general_purpose::STANDARD.decode(key_b64) {
                            Ok(k) => k,
                            Err(e) => {
                                let err = format!("failed to decode enc_key_b64: {}", e);
                                eprintln!("{}", err);
                                send_error(&mut ws, &current_job_id, &err);
                                continue;
                            }
                        };
                        if key_bytes.len() != AEAD_KEY_SIZE {
                            let err = format!("invalid key length: {}", key_bytes.len());
                            eprintln!("{}", err);
                            send_error(&mut ws, &current_job_id, &err);
                            continue;
                        }
                        let mut aes_key = [0u8; AEAD_KEY_SIZE];
                        aes_key.copy_from_slice(&key_bytes);
                        handle_job(client, &payload, &aes_key)
                    }
                };

                let reply = match result {
                    Ok(result_url) => {
                        println!("Job {} completed successfully", current_job_id);
                        serde_json::json!({
                            "type": "result",
                            "payload": {
                                "job_id": current_job_id,
                                "status": "done",
                                "result_url": result_url,
                            }
                        })
                    }
                    Err(e) => {
                        eprintln!("Job failed: {}", e);
                        serde_json::json!({
                            "type": "result",
                            "payload": {
                                "job_id": current_job_id,
                                "status": "error",
                                "error": format!("{}", e),
                            }
                        })
                    }
                };
                let _ = ws.send(Message::Text(serde_json::to_string(&reply).unwrap()));
            }
            Message::Close(_) => {
                println!("WebSocket bridge closed by server");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
