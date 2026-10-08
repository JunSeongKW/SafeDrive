"""Record official dataset files and authenticated access without accepting terms."""
import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, get_token
import requests

ROOT = Path(__file__).resolve().parents[1]


def main(arguments):
    api = HfApi()
    token = get_token()
    dojo_repositories = sorted({info.id for info in api.list_datasets(author="Yuqi1997", search="DrivingDojo")
                                if info.id.split("/")[-1] == "DrivingDojo" or info.id.split("/")[-1].startswith("DrivingDojo-Extra")})
    repositories = ["turing-motors/CoVLA-Dataset", *dojo_repositories, "OpenDriveLab/OpenScene"]
    records = []
    for repository in repositories:
        info = api.dataset_info(repository, files_metadata=True, timeout=30)
        files = []
        for item in info.siblings:
            name = item.rfilename
            if "CoVLA" in repository:
                keep = name.endswith(".mp4")
            elif "DrivingDojo" in repository:
                keep = "video" in name and name.endswith((".tar.gz", ".tgz", ".tar", ".zip"))
            else:
                keep = "openscene-v1.1/openscene_sensor_trainval_camera/" in name
            if keep:
                files.append({"filename": name, "bytes": item.size, "lfs_sha256": item.lfs.sha256 if item.lfs else None})
        assert files, repository
        sample_url = "https://huggingface.co/datasets/" + repository + "/resolve/" + info.sha + "/" + files[0]["filename"]
        response = requests.head(sample_url, headers={"Authorization": "Bearer " + token} if token else {},
                                 allow_redirects=True, timeout=30)
        record = {"repository": repository, "revision": info.sha, "gated": info.gated,
            "access_status": response.status_code, "error_code": response.headers.get("X-Error-Code"),
            "error_message": response.headers.get("X-Error-Message"), "files": files,
            "total_source_bytes": sum(item["bytes"] or 0 for item in files)}
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "files"}), flush=True)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_unix": time.time(), "token_present": bool(token), "records": records,
        "all_sources_accessible": all(record["access_status"] == 200 for record in records),
        "terms_acceptance_automated": False, "exact_drive_jepa_330h_manifest_available": False}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/download_catalog.json")
    main(parser.parse_args())
