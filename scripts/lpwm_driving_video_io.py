"""Verified, resumable HTTP reads for the authorized driving-video corpus."""
import hashlib
import io
import time

import requests
from urllib3.exceptions import HTTPError as TransportError


class ResumableVerifiedReader(io.RawIOBase):
    def __init__(self, url, headers, expected_bytes, expected_sha256, progress=None):
        super().__init__()
        self.url = url
        self.headers = headers
        self.expected_bytes = expected_bytes
        self.expected_sha256 = expected_sha256
        self.progress = progress
        self.byte_count = 0
        self.digest = hashlib.sha256()
        self.response = None
        self.reconnections = 0
        self.last_progress = 0.

    def readable(self):
        return True

    def reconnect(self):
        if self.response is not None:
            self.response.close()
        headers = dict(self.headers)
        if self.byte_count:
            headers["Range"] = f"bytes={self.byte_count}-"
        self.response = requests.get(self.url, headers=headers, stream=True, timeout=(30, 90))
        self.response.raise_for_status()
        if self.byte_count:
            assert self.response.status_code == 206, "Server must honor the resume byte range"
            assert self.response.headers["Content-Range"].startswith(f"bytes {self.byte_count}-")
        elif self.response.status_code == 206:
            assert self.response.headers["Content-Range"].startswith("bytes 0-")

    def read(self, size=-1):
        if self.byte_count == self.expected_bytes:
            return b""
        requested = self.expected_bytes - self.byte_count if size < 0 else min(size, self.expected_bytes - self.byte_count)
        if requested == 0:
            return b""
        for attempt in range(8):
            try:
                if self.response is None:
                    self.reconnect()
                content = self.response.raw.read(requested)
                if not content:
                    raise ConnectionError("Unexpected HTTP EOF before the pinned file size")
                self.byte_count += len(content)
                self.digest.update(content)
                if self.progress and time.time() - self.last_progress >= 20:
                    self.progress(self.byte_count, self.reconnections)
                    self.last_progress = time.time()
                return content
            except (requests.RequestException, TransportError, ConnectionError) as error:
                if isinstance(error, requests.HTTPError) and error.response.status_code in (401, 403, 404):
                    raise
                if self.response is not None:
                    self.response.close()
                self.response = None
                self.reconnections += 1
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))

    def verify(self):
        while self.read(1024 * 1024):
            pass
        assert self.byte_count == self.expected_bytes
        assert self.digest.hexdigest() == self.expected_sha256, "Source checksum mismatch"

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()
