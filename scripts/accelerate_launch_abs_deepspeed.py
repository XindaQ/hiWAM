#!/usr/bin/env python3
"""Run accelerate launch while resolving the deepspeed CLI to an absolute path."""

import os
import runpy
import shutil
import subprocess
import sys


def main() -> None:
    deepspeed_bin = os.environ.get("DEEPSPEED_BIN") or shutil.which("deepspeed")
    if not deepspeed_bin:
        raise FileNotFoundError("deepspeed executable was not found on PATH")

    fastwam_env = os.environ.get("FASTWAM_ENV")
    env_bin = os.path.dirname(deepspeed_bin)
    path_prefixes = [env_bin]
    if fastwam_env:
        path_prefixes.insert(0, os.path.join(fastwam_env, "bin"))

    parent_path = os.environ.get("PATH", "")
    patched_path = ":".join(path_prefixes + [parent_path])
    os.environ["PATH"] = patched_path

    original_popen = subprocess.Popen

    def patched_popen(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "deepspeed":
            cmd = list(cmd)
            cmd[0] = deepspeed_bin

        child_env = kwargs.get("env")
        if child_env is not None:
            child_env = dict(child_env)
            child_env["PATH"] = ":".join(path_prefixes + [child_env.get("PATH", parent_path)])
            kwargs["env"] = child_env

        return original_popen(cmd, *args, **kwargs)

    subprocess.Popen = patched_popen
    print(f"[accelerate_wrapper] deepspeed_bin={deepspeed_bin}", flush=True)
    print(f"[accelerate_wrapper] PATH={os.environ.get('PATH')}", flush=True)

    sys.argv = ["accelerate.commands.launch", *sys.argv[1:]]
    runpy.run_module("accelerate.commands.launch", run_name="__main__")


if __name__ == "__main__":
    main()
