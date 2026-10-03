"""Download the public Sketchy checkpoint linked by LPWM; verify MEGA file MAC."""
import base64
import hashlib
import json
import struct
from pathlib import Path

import requests
from Crypto.Cipher import AES


def download_checkpoint():
    artifact_root = Path(__file__).resolve().parents[1] / "outputs/lpwm_navsim_adaptation_v1/pretrained"
    artifact_root.mkdir(parents=True, exist_ok=True)
    public_key = "ndHMuT0ifI3xEFG_PCa6AX0BP_e8vM8hRJHZ_lBNQE8"
    key_words = struct.unpack(">8I", base64.urlsafe_b64decode(public_key + "="))
    aes_key = struct.pack(">4I", *(key_words[index] ^ key_words[index + 4] for index in range(4)))
    response = requests.post("https://g.api.mega.co.nz/cs", params={"id": 234567},
                             json=[{"a": "g", "g": 1, "p": "YVMnyZYJ"}], timeout=30)
    response.raise_for_status()
    metadata = response.json()[0]
    assert isinstance(metadata, dict) and 0 < metadata["s"] < 1024**3, metadata
    attributes = base64.urlsafe_b64decode(metadata["at"] + "=" * (-len(metadata["at"]) % 4))
    attributes = AES.new(aes_key, AES.MODE_CBC, iv=bytes(16)).decrypt(attributes).rstrip(b"\0")
    assert attributes.startswith(b"MEGA")
    filename = json.loads(attributes[4:])["n"]
    assert Path(filename).name == filename
    destination = artifact_root / filename
    if destination.exists():
        raise FileExistsError(destination)
    pending = destination.with_suffix(destination.suffix + ".partial")
    decryptor = AES.new(aes_key, AES.MODE_CTR, nonce=b"", initial_value=struct.pack(">4I", *key_words[4:6], 0, 0))
    downloaded_bytes = 0
    with requests.get(metadata["g"], stream=True, timeout=(30, 120)) as download, pending.open("wb") as output:
        download.raise_for_status()
        for encrypted_chunk in download.iter_content(1024**2):
            output.write(decryptor.decrypt(encrypted_chunk))
            downloaded_bytes += len(encrypted_chunk)
    assert downloaded_bytes == metadata["s"]
    chunk_iv = struct.pack(">4I", *key_words[4:6], *key_words[4:6])
    accumulator = bytes(16)
    ecb_cipher = AES.new(aes_key, AES.MODE_ECB)
    digest = hashlib.sha256()
    with pending.open("rb") as source:
        chunk_index = 1
        while plaintext := source.read(min(chunk_index, 8) * 128 * 1024):
            digest.update(plaintext)
            padded = plaintext + bytes(-len(plaintext) % 16)
            chunk_mac = AES.new(aes_key, AES.MODE_CBC, iv=chunk_iv).encrypt(padded)[-16:]
            accumulator = ecb_cipher.encrypt(bytes(first ^ second for first, second in zip(accumulator, chunk_mac)))
            chunk_index += 1
    mac_words = struct.unpack(">4I", accumulator)
    assert (mac_words[0] ^ mac_words[1], mac_words[2] ^ mac_words[3]) == key_words[6:8], "MEGA MAC mismatch"
    pending.rename(destination)
    provenance = {"official_source": "https://github.com/taldatech/lpwm", "public_handle": "YVMnyZYJ",
                  "filename": filename, "bytes": downloaded_bytes, "sha256": digest.hexdigest(), "mega_mac_verified": True}
    (artifact_root / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    download_checkpoint()
