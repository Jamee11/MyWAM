#!/usr/bin/env python3
"""Read-only GPU access check from a new process; no model or tensor allocation."""
import ctypes
import json
import os
import subprocess
import sys


def check():
    report = {}
    for path in ['/dev/nvidiactl', '/dev/nvidia0', '/dev/nvidia-uvm']:
        try:
            descriptor = os.open(path, os.O_RDWR)
            os.close(descriptor)
            report[path] = 'open_ok'
        except OSError as error:
            report[path] = dict(errno=error.errno, error=error.strerror)
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=15)
    report['nvml_exit_code'] = result.returncode
    report['nvml_output'] = (result.stdout + result.stderr).strip()
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        code = driver.cuInit(0)
        name = ctypes.c_char_p()
        driver.cuGetErrorName(code, ctypes.byref(name))
        report['fresh_cuda_init'] = dict(code=code, name=name.value.decode() if name.value else None)
    except OSError as error:
        code = -1
        report['fresh_cuda_init'] = dict(code=code, error=str(error))
    passed = result.returncode == 0 and code == 0 and all(
        report[path] == 'open_ok' for path in ['/dev/nvidiactl', '/dev/nvidia0', '/dev/nvidia-uvm'])
    report['status'] = 'passed' if passed else 'failed_do_not_switch_training_yet'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(check())
