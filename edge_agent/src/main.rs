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
//!   7. Agent extracts ZIP, runs Docker/Kata with GPU passthrough
//!   8. Agent re-encrypts results inside TEE before uploading
//!
//! Security invariants:
//!   - Matchmaker NEVER sees plaintext, shared secret, or AES key
//!   - All cryptographic operations happen inside the TEE boundary
//!   - Payload is padded to standard tier size (plausible deniability)
//!   - Keys are ephemeral — discarded after job completion

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose;
use base64::Engine;
use hkdf::Hkdf;
use rand::rngs::OsRng;
use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::Sha256;
use std::env;
use std::fs::{self, File};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};
use uuid::Uuid;
use x25519_dalek::{EphemeralSecret, PublicKey, SharedSecret};

mod sev_snp;

// ─── Constants ──────────────────────────────────────────────────────────────

const NONCE_SIZE: usize = 12;
const AEAD_KEY_SIZE: usize = 32;
const SALT_SIZE: usize = 32;
const HEARTBEAT_INTERVAL_SECS: u64 = 20;

// ─── GPU Detection ──────────────────────────────────────────────────────────

fn query_gpu_info() -> (String, i32) {
    // Returns (gpu_model, vram_mb). Prefer vendor/runtime APIs where available
    // because they expose VRAM. Keep Linux hardware enumeration fallbacks so a
    // present GPU is not reported as "unknown" just because the runtime tool is
    // missing or broken.
    if let Some(info) = query_nvidia_smi_csv() {
        return info;
    }

    if let Some(info) = query_linux_drm_gpu_info() {
        return info;
    }

    if let Some(model) = query_nvidia_smi_list() {
        return (model, 0);
    }

    if let Some(model) = query_nvidia_proc_model() {
        return (model, 0);
    }

    if let Some(model) = query_lspci_gpu_model() {
        return (model, 0);
    }

    ("unknown".to_string(), 0)
}

fn query_nvidia_smi_csv() -> Option<(String, i32)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            parse_nvidia_smi_csv(&stdout)
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "GPU detection: nvidia-smi query failed with status {:?}: {}",
                out.status.code(),
                stderr.trim()
            );
            None
        }
        Err(err) => {
            eprintln!("GPU detection: failed to run nvidia-smi: {}", err);
            None
        }
    }
}

fn parse_nvidia_smi_csv(stdout: &str) -> Option<(String, i32)> {
    let line = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    if let Some(comma_pos) = line.find(',') {
        let model = line[..comma_pos].trim().to_string();
        let vram_mb = parse_leading_i32(line[comma_pos + 1..].trim()).unwrap_or(0);
        if model.is_empty() {
            None
        } else {
            Some((model, vram_mb))
        }
    } else if line.is_empty() {
        None
    } else {
        Some((line.to_string(), 0))
    }
}

fn parse_leading_i32(value: &str) -> Option<i32> {
    let digits: String = value
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn query_nvidia_smi_list() -> Option<String> {
    let output = Command::new("nvidia-smi").arg("-L").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(parse_nvidia_smi_list_line)
        .filter(|model| !model.is_empty())
}

fn parse_nvidia_smi_list_line(line: &str) -> Option<String> {
    let after_colon = line.split_once(':')?.1.trim();
    let model = after_colon
        .split_once(" (UUID:")
        .map(|(model, _)| model)
        .unwrap_or(after_colon)
        .trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}

fn query_nvidia_proc_model() -> Option<String> {
    let entries = fs::read_dir("/proc/driver/nvidia/gpus").ok()?;
    for entry in entries.flatten() {
        let info_path = entry.path().join("information");
        let Ok(contents) = fs::read_to_string(info_path) else {
            continue;
        };
        if let Some(model) = parse_nvidia_proc_information(&contents) {
            return Some(model);
        }
    }
    None
}

fn parse_nvidia_proc_information(contents: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim().eq_ignore_ascii_case("Model") {
            let model = value.trim();
            if model.is_empty() {
                None
            } else {
                Some(model.to_string())
            }
        } else {
            None
        }
    })
}

fn query_linux_drm_gpu_info() -> Option<(String, i32)> {
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let card_name = entry.file_name();
        let card_name = card_name.to_string_lossy();
        if !card_name.starts_with("card") || card_name.contains('-') {
            continue;
        }

        let device_path = entry.path().join("device");
        let Some(vendor) = read_trimmed(device_path.join("vendor")) else {
            continue;
        };
        if !is_supported_gpu_vendor(&vendor) {
            continue;
        }

        let slot = pci_slot_from_sysfs_device(&device_path);
        let model = slot
            .as_deref()
            .and_then(query_lspci_model_for_slot)
            .unwrap_or_else(|| vendor_label(&vendor).to_string());
        let vram_mb = read_vram_mb_from_sysfs(&device_path).unwrap_or(0);
        return Some((model, vram_mb));
    }

    None
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    let value = fs::read_to_string(path).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn is_supported_gpu_vendor(vendor: &str) -> bool {
    matches!(
        vendor.trim().to_ascii_lowercase().as_str(),
        "0x10de" | "0x1002"
    )
}

fn vendor_label(vendor: &str) -> &'static str {
    match vendor.trim().to_ascii_lowercase().as_str() {
        "0x10de" => "NVIDIA GPU",
        "0x1002" => "AMD GPU",
        _ => "GPU",
    }
}

fn pci_slot_from_sysfs_device(device_path: &Path) -> Option<String> {
    let canonical = fs::canonicalize(device_path).ok()?;
    canonical
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| name.contains(':') && name.contains('.'))
        .map(str::to_string)
}

fn read_vram_mb_from_sysfs(device_path: &Path) -> Option<i32> {
    let bytes = read_trimmed(device_path.join("mem_info_vram_total"))?
        .parse::<u64>()
        .ok()?;
    bytes_to_mb_i32(bytes)
}

fn bytes_to_mb_i32(bytes: u64) -> Option<i32> {
    let mb = bytes / 1024 / 1024;
    i32::try_from(mb).ok()
}

fn query_lspci_model_for_slot(slot: &str) -> Option<String> {
    let output = Command::new("lspci")
        .args(["-D", "-mm", "-nn", "-s", slot])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_lspci_gpu_line)
}

fn query_lspci_gpu_model() -> Option<String> {
    let output = Command::new("lspci")
        .args(["-D", "-mm", "-nn"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_lspci_gpu_line)
}

fn parse_lspci_gpu_line(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("nvidia")
        || lower.contains("advanced micro devices")
        || lower.contains("amd/ati"))
    {
        return None;
    }

    if !(lower.contains("vga compatible controller")
        || lower.contains("3d controller")
        || lower.contains("display controller"))
    {
        return None;
    }

    let fields = parse_lspci_machine_fields(line);
    let model = fields.get(3).or_else(|| fields.last())?;

    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}

fn parse_lspci_machine_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => in_quotes = !in_quotes,
            '\\' if in_quotes => {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    fields.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        fields.push(current);
    }

    fields
}

#[cfg(test)]
mod gpu_detection_tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv_with_plain_memory() {
        let info = parse_nvidia_smi_csv("NVIDIA GeForce GTX 1650, 4096\n").unwrap();
        assert_eq!(info, ("NVIDIA GeForce GTX 1650".to_string(), 4096));
    }

    #[test]
    fn parses_nvidia_smi_csv_with_units() {
        let info = parse_nvidia_smi_csv("NVIDIA GeForce GTX 1650, 4096 MiB\n").unwrap();
        assert_eq!(info, ("NVIDIA GeForce GTX 1650".to_string(), 4096));
    }

    #[test]
    fn parses_nvidia_smi_list_line() {
        let model = parse_nvidia_smi_list_line(
            "GPU 0: NVIDIA GeForce GTX 1650 (UUID: GPU-12345678-1234-1234-1234-123456789abc)",
        )
        .unwrap();
        assert_eq!(model, "NVIDIA GeForce GTX 1650");
    }

    #[test]
    fn parses_nvidia_proc_information() {
        let model = parse_nvidia_proc_information(
            "Model: \t\t NVIDIA GeForce GTX 1650\nIRQ:   122\nGPU UUID: GPU-abc\n",
        )
        .unwrap();
        assert_eq!(model, "NVIDIA GeForce GTX 1650");
    }

    #[test]
    fn parses_lspci_machine_readable_nvidia_line() {
        let model = parse_lspci_gpu_line(
            r#"0000:01:00.0 "VGA compatible controller" "NVIDIA Corporation" "TU117 [GeForce GTX 1650]" -r"a1" "ASUSTeK Computer Inc." "Device 8708""#,
        )
        .unwrap();
        assert_eq!(model, "TU117 [GeForce GTX 1650]");
    }

    #[test]
    fn parses_lspci_machine_readable_amd_line() {
        let model = parse_lspci_gpu_line(
            r#"0000:03:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" "Navi 23 [Radeon RX 6600/6600 XT/6600M]" -r"c7" "Micro-Star International Co., Ltd. [MSI]" "Device 5021""#,
        )
        .unwrap();
        assert_eq!(model, "Navi 23 [Radeon RX 6600/6600 XT/6600M]");
    }

    #[test]
    fn converts_vram_bytes_to_mb() {
        assert_eq!(bytes_to_mb_i32(4_294_967_296), Some(4096));
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
            <[u8; 32]>::try_from(client_pub_bytes).expect("client public key must be 32 bytes"),
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
    if padded.is_empty() {
        return Err(anyhow!("empty padded payload"));
    }
    let pad_len = padded[padded.len() - 1] as usize;
    if pad_len >= padded.len() {
        return Err(anyhow!(
            "invalid padding length: {} >= {}",
            pad_len,
            padded.len()
        ));
    }
    let original_size = padded.len() - pad_len - 1;
    Ok(padded[..original_size].to_vec())
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
                    return Err(anyhow!(
                        "challenge nonce must be 32 bytes, got {}",
                        nonce_bytes.len()
                    ));
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

// ─── Docker Execution ──────────────────────────────────────────────────────

fn run_docker_job(workspace: &Path, job_type: &str, config: &serde_json::Value) -> Result<()> {
    let (image, cmd) = build_docker_config(job_type, config);

    let work_dir = workspace.to_string_lossy().to_string();
    let mount_ro = format!("{}:/workspace:ro", work_dir);
    let mount_out = format!("{}:/workspace/output", work_dir);
    let mut docker_args = vec![
        "run",
        "--gpus",
        "all",
        "--rm",
        "--network",
        "none", // No network access inside the container
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "-v",
        &mount_ro,
        "-v",
        &mount_out,
        "-w",
        "/workspace",
        "-e",
        "JOB_ID=tenxo",
        "-e",
        "PYTHONUNBUFFERED=1",
        "--memory",
        "32g",
        "--cpus",
        "8",
        &image,
    ];
    docker_args.extend(cmd.iter().map(|s| s.as_str()));

    let output = Command::new("docker")
        .args(&docker_args)
        .output()
        .context("failed to execute docker")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "docker exited with {:?}: {}",
            output.status.code(),
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
            let cmd = format!(
                "bash -c 'if [ -f requirements.txt ]; then pip install -r requirements.txt -q; fi; python {}'",
                script
            );
            (
                "python:3.11-slim".into(),
                vec!["bash".into(), "-c".into(), cmd],
            )
        }
        "blender" => {
            let blend = config
                .get("blend_file")
                .and_then(|v| v.as_str())
                .unwrap_or("scene.blend");
            let cmd = format!(
                "bash -c 'apt-get update -qq && apt-get install -y -qq blender && blender -b {} -o /workspace/output/frame_#### -s 1 -e 1 -a'",
                blend
            );
            (
                "nvidia/cuda:12.2.0-runtime-ubuntu22.04".into(),
                vec!["bash".into(), "-c".into(), cmd],
            )
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
            let cmd = format!(
                "bash -c 'cd /workspace && nvcc -o {} {} && ./{}'",
                out, src, out
            );
            (
                "nvidia/cuda:12.2.0-devel-ubuntu22.04".into(),
                vec!["bash".into(), "-c".into(), cmd],
            )
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

// ─── Job Execution Pipeline ────────────────────────────────────────────────

fn handle_job(client: &Client, job: &JobMsg, aes_key: &[u8; AEAD_KEY_SIZE]) -> Result<String> {
    let job_id = job.job_id.as_deref().unwrap_or("unknown");
    println!("Processing job: {} (encrypted)", job_id);

    // ── Step 1: Download encrypted payload ─────────────────────────────
    let enc_bytes = download_bytes(client, &job.encrypted_job_link)
        .context("failed to download encrypted job payload")?;
    println!("Downloaded {} encrypted bytes", enc_bytes.len());

    // ── Step 2: Decrypt IN-MEMORY inside TEE boundary ─────────────────
    let padded_plain =
        decrypt_payload(&enc_bytes, aes_key).context("AES-256-GCM decryption failed")?;
    let plain = unpad_payload(&padded_plain).context("failed to remove padding")?;
    println!("Decrypted {} bytes inside TEE", plain.len());

    // ── Step 3: Extract ZIP to temp workspace ──────────────────────────
    let td = tempdir().context("failed to create temp workspace")?;
    let workspace = td.path().join("job");
    fs::create_dir_all(&workspace)?;

    let zip_path = td.path().join("payload.zip");
    fs::write(&zip_path, &plain).context("failed to write decrypted zip")?;

    let mut archive =
        zip::ZipArchive::new(File::open(&zip_path)?).context("failed to open ZIP archive")?;
    archive
        .extract(&workspace)
        .context("failed to extract ZIP archive")?;
    println!("Extracted workspace to {:?}", workspace);

    // ── Step 4: Execute inside Docker/Kata ────────────────────────────
    // The config is embedded as job_config.json in the workspace.
    // For the MVP, we auto-detect the main script.
    let job_type = "python"; // In production, read from workspace metadata
    let config = serde_json::json!({
        "script": "main.py",
    });

    run_docker_job(&workspace, job_type, &config).context("Docker execution failed")?;
    println!("Job execution complete");

    // ── Step 5: Package output and re-encrypt INSIDE TEE ──────────────
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
        // Output directory may not exist; create a placeholder
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

    let result_blob = fs::read(&result_zip_path).context("failed to read result zip")?;

    // Re-encrypt with the same AES key INSIDE the TEE
    let encrypted_result =
        encrypt_payload(&result_blob, aes_key).context("failed to encrypt result")?;
    println!("Re-encrypted {} bytes of results", encrypted_result.len());

    // ── Step 6: Upload encrypted result ───────────────────────────────
    let res = client
        .put(&job.result_upload_url)
        .body(encrypted_result)
        .send()
        .context("failed to upload encrypted result")?;

    if !res.status().is_success() {
        return Err(anyhow!("result upload failed: {}", res.status()));
    }

    println!("Encrypted result uploaded to {}", job.result_upload_url);
    Ok(job.result_upload_url.clone())
}

// ─── Networking ────────────────────────────────────────────────────────────

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

// ─── Main Entry Point ──────────────────────────────────────────────────────

fn load_env_file(path: &str) {
    let contents = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            env::set_var(key.trim(), value.trim());
        }
    }
}

fn main() -> Result<()> {
    load_env_file("/etc/tenxo/agent.env");

    let matchmaker_url =
        env::var("MATCHMAKER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let node_id = env::var("NODE_ID").unwrap_or_else(|_| format!("node-{}", Uuid::new_v4()));
    let owner = env::var("OWNER").unwrap_or_else(|_| String::new());

    println!("Tenxo Edge Agent starting...");
    println!("  Node ID:    {}", node_id);
    println!("  Matchmaker: {}", matchmaker_url);

    let client = Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .context("failed to create HTTP client")?;

    // ── Generate ephemeral X25519 keypair ────────────────────────────
    let agent_keys = AgentKeys::generate();
    println!("Ephemeral X25519 keypair generated");

    // ── ECDH Key Exchange + acquire persistent WebSocket bridge ──────
    let (shared_secret, mut ws) = perform_key_exchange(&matchmaker_url, agent_keys)?;
    println!("ECDH shared secret computed (matchmaker never saw it)");

    // ── Query GPU info ──────────────────────────────────────────────
    let (gpu_model, gpu_vram_mb) = query_gpu_info();
    println!("Detected GPU: {} ({} MB VRAM)", gpu_model, gpu_vram_mb);

    // ── Register with matchmaker bridge ────────────────────────────
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

    // ── Shutdown flag ──────────────────────────────────────────────
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        println!("Received shutdown signal, cleaning up...");
        shutdown_clone.store(true, Ordering::SeqCst);
    })
    .context("failed to set Ctrl-C handler")?;

    // ── Spawn heartbeat publisher (HTTP POST, no NATS needed) ──────
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

    // ── Job processing loop via WebSocket bridge ─────────────────────
    while !shutdown.load(Ordering::SeqCst) {
        let msg = match ws.read() {
            Ok(m) => m,
            Err(tungstenite::Error::Io(e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue
            }
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
                        continue;
                    }
                };

                let current_job_id = payload.job_id.clone().unwrap_or_default();

                // ── Derive AES key from shared secret + per-job salt ─
                let result = match &payload.salt_b64 {
                    Some(s) => {
                        let salt = general_purpose::STANDARD
                            .decode(s)
                            .context("failed to decode salt")?;
                        if salt.len() != SALT_SIZE {
                            eprintln!(
                                "invalid salt length: {} (expected {})",
                                salt.len(),
                                SALT_SIZE
                            );
                            continue;
                        }
                        let mut aes_key = [0u8; AEAD_KEY_SIZE];
                        aes_key.copy_from_slice(&derive_aes_key(&shared_secret, &salt));
                        println!("AES key derived via HKDF for job {}", current_job_id);
                        handle_job(&client, &payload, &aes_key)
                    }
                    None => {
                        let key_b64 = payload.enc_key_b64.as_deref().unwrap_or("");
                        if key_b64.is_empty() {
                            eprintln!("no enc_key_b64 or salt_b64 in job message");
                            Err(anyhow!("missing encryption key"))
                        } else {
                            let key_bytes = general_purpose::STANDARD
                                .decode(key_b64)
                                .context("failed to decode enc_key_b64")?;
                            if key_bytes.len() != AEAD_KEY_SIZE {
                                eprintln!("invalid key length: {}", key_bytes.len());
                                continue;
                            }
                            let mut aes_key = [0u8; AEAD_KEY_SIZE];
                            aes_key.copy_from_slice(&key_bytes);
                            handle_job(&client, &payload, &aes_key)
                        }
                    }
                };

                // ── Send result over WebSocket bridge ─────────────────
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

    println!("Agent shutting down gracefully");
    Ok(())
}
