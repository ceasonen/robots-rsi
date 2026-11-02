"""Generic owned policy sidecar; no simulator or device initialization."""

import json
import queue
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PolicyPipe:
    def __init__(self, python, worker, work):
        assert python.is_relative_to(ROOT) and worker.is_relative_to(ROOT) and work.is_relative_to(ROOT)
        work.mkdir(parents=True, exist_ok=True)
        self.log = (work / "worker.stderr.log").open("a", encoding="utf8")
        self.process = subprocess.Popen([str(python), "-u", str(worker)], cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf8", bufsize=1)
        self.lines = queue.Queue()
        def drain():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=drain, daemon=True).start()
        try:
            response = self.receive(600)
            assert response["ready"]
            self.metadata, self.worker_pid = response["metadata"], response["pid"]
        except BaseException:
            self.terminate()
            self.log.close()
            raise

    def receive(self, timeout=180):
        line = self.lines.get(timeout=timeout)
        assert line is not None, "Owned policy worker ended; inspect stderr"
        result = json.loads(line)
        assert "error" not in result, result
        return result

    def request(self, command, **arguments):
        self.process.stdin.write(json.dumps({"command": command, **arguments}) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def terminate(self):
        if self.process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True, check=False)
            self.process.wait(timeout=30)

    def close(self):
        try:
            if self.process.poll() is None:
                self.request("close")
                self.process.wait(timeout=30)
        finally:
            self.terminate()
            self.log.close()
