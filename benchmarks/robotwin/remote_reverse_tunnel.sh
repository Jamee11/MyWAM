#!/usr/bin/env bash
# Run on MODEL host; expose its loopback model slots on the SIMULATOR host.
# Usage: remote_reverse_tunnel.sh user@sim-host ssh-port sim-port:model-port [...]
set -euo pipefail
if (( $# < 3 )); then
    echo 'Usage: remote_reverse_tunnel.sh user@sim-host ssh-port sim-port:model-port [...]' >&2
    exit 2
fi
target="$1"
ssh_port="$2"
shift 2
[[ "$target" != -* && "$ssh_port" =~ ^[0-9]+$ ]] || exit 2
# Background SSH must never request terminal input. Configure keys/known_hosts first.
args=(-n -N -T -p "$ssh_port" -o BatchMode=yes -o ConnectTimeout=15 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
for mapping in "$@"; do
    [[ "$mapping" =~ ^([0-9]+):([0-9]+)$ ]] || exit 2
    sim_port="${BASH_REMATCH[1]}"
    model_port="${BASH_REMATCH[2]}"
    (( sim_port >= 1 && sim_port <= 65535 && model_port >= 1 && model_port <= 65535 )) || exit 2
    args+=(-R "127.0.0.1:${sim_port}:127.0.0.1:${model_port}")
done
exec ssh "${args[@]}" "$target"
