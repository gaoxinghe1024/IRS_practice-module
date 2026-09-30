"""Use Windows system proxy/TLS settings when available; standard urllib elsewhere."""
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


def download_file(url, destination, timeout=120):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shell = (shutil.which("pwsh") or shutil.which("powershell")) if os.name == "nt" else None
    if shell:
        try:
            result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-File", str(Path(__file__).with_name("fetch.ps1")),
                                     "-Url", url, "-OutFile", str(destination), "-TimeoutSec", str(timeout)],
                                    capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired as error:
            raise OSError("HTTP transfer timeout") from error
        if result.returncode:
            raise OSError(result.stderr.decode("utf-8", errors="replace")[-1200:])
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "AnimeFinder-course-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as stream:
            shutil.copyfileobj(response, stream)
