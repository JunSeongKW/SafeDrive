"""Download the two public pre-driving backbones and record provenance."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'runtime/checkpoints/four_model_small_corpus'
SOURCES = {
    'vjepa2_vitl.pt': 'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt',
    'dinov2_vits_reg4.safetensors': 'https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m/resolve/main/model.safetensors',
}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = DESTINATION / name
        if target.exists() and target.with_suffix(target.suffix + '.json').exists():
            continue
        pending = target.with_suffix(target.suffix + '.downloading')
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'research-backbone-download'})
                with urllib.request.urlopen(request, timeout=120) as response, pending.open('wb') as stream:
                    size = int(response.headers.get('Content-Length', 0))
                    received, updated = 0, 0
                    sha = hashlib.sha256()
                    while chunk := response.read(8 * 1024**2):
                        stream.write(chunk)
                        sha.update(chunk)
                        received += len(chunk)
                        if time.time() - updated > 15:
                            print(json.dumps({'name': name, 'received': received, 'expected': size}), flush=True)
                            updated = time.time()
                assert not size or received == size
                pending.replace(target)
                target.with_suffix(target.suffix + '.json').write_text(json.dumps({
                    'source_url': url, 'bytes': received, 'sha256': sha.hexdigest(),
                    'downloaded_unix': time.time(), 'driving_adapted_weights': False,
                }, indent=2) + '\n')
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10)


if __name__ == '__main__':
    main()
