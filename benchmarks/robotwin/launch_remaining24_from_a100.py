"""Start the shared 24-condition dispatcher on 4090 through SSH, from A100."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--sim-gpus", default="0,1,2,4,5,6")
    p.add_argument("--ports", default="22700,22701,22702,22710,22711,22712")
    p.add_argument("--ssh-host", default="root@210.45.70.21")
    p.add_argument("--ssh-port", type=int, default=26199)
    p.add_argument("--sim-python", type=Path, default=ROOT / ".conda/robotwin-official/bin/python")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run = args.model_run.absolute()
    output = args.output.absolute() if args.output else run / "remaining24_4090"
    command = [
        str(args.sim_python.absolute()),
        "-u",
        str(ROOT / "benchmarks/robotwin/eval_remaining24.py"),
        "--model-run",
        str(run),
        "--output",
        str(output),
        "--sim-gpus",
        args.sim_gpus,
        "--ports",
        args.ports,
    ]
    # All argv values are shell-quoted before being transported by OpenSSH.
    bootstrap = """import json,os,subprocess,sys
from pathlib import Path
root,run,out=sys.argv[1:4];command=sys.argv[4:]
os.chdir(root)
out=Path(out);run=Path(run)
marker=out.with_name(out.name+"_launch.json")
if out.exists():raise SystemExit("Batch output exists: choose a new output directory")
with marker.open("x") as m:
 with out.with_name(out.name+"_dispatcher.log").open("w") as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=child.pid,host=os.uname().nodename,command=command,output=str(out))
 m.write(json.dumps(record,indent=2)+"\\n")
print(json.dumps(record,indent=2))
"""
    remote = shlex.join([str(args.sim_python.absolute()), "-c", bootstrap, str(ROOT), str(run), str(output), *command])
    ssh = [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(args.ssh_port),
        args.ssh_host,
        remote,
    ]
    if args.dry_run:
        print(
            json.dumps(
                dict(simulator_command=command, ssh_host=args.ssh_host, ssh_port=args.ssh_port, output=str(output)),
                indent=2,
            )
        )
        return 0
    return subprocess.call(ssh)


if __name__ == "__main__":
    raise SystemExit(main())
