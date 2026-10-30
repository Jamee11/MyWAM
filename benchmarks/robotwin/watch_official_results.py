"""Maintain a local live analysis and record newly completed experimental results."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/robotwin_eval"


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    baseline = BASE / "baseline_official_manifest_deadline_20261018"
    sidecar = BASE / "sidecar20000_officialenv_20261018"
    memory_path = sidecar / "analysis_monitor.json"
    memory = read(memory_path, {"recorded": [], "history": []})
    historical = read(BASE / "parallel_official100_20261016/parallel.json")
    old = {(j["task"], j["mode"]): j["result"]["successes"] for j in historical["jobs"] if j["status"] == "complete"}
    for result in read(BASE / "historical_recovery_stack_bowls_three_randomized_20261018/run.json")["results"]:
        old[(result["task"], result["mode"])] = result["successes"]
    assert len(old) == 100
    while True:
        now = datetime.now(timezone.utc)
        state = read(baseline / "run.json", {})
        lines = ["# RoboTwin 新环境实时结果分析", "", f"更新时间：{now.isoformat()}", "",
                 "原版传统官方任务评测继续；并行组复用官方固定manifest协议（seed/指令/CUDA RNG固定），两者分别记录。",
                 "每项100条才视为任务完整结果；进行中的百分比不作为全任务宏平均。", "",
                 "## 原版异常任务并行复测", "", "| 任务 | 条件 | 专家清单 | 策略成功/完成 | 旧环境百条 |",
                 "|---|---|---:|---:|---:|"]
        replacements = {("place_dual_shoes", "demo_clean"): 92}
        new_baseline = dict(replacements)
        valid_shards = {(r["task"], r["mode"], r["episode_start"]) for r in state.get("results", [])}
        total = 0
        for case in state.get("cases", []):
            task, mode = case["task"], case["mode"]
            manifest = read(baseline / "manifests" / mode / task / "progress.json", {})
            count = success = verified = 0
            for start in range(0, 100, 10):
                directory = baseline / "shards" / mode / task / f"{start:03d}"
                result = read(directory / "result.json")
                progress = read(directory / "progress.json", {})
                if result:
                    count += result["num_episodes"]
                    success += result["successes"]
                    if (task, mode, start) in valid_shards:
                        verified += result["num_episodes"]
                else:
                    count += progress.get("completed", 0)
                    success += progress.get("successes", 0)
            total += count
            lines.append(f"| {task} | {mode} | {manifest.get('completed', 0)}/100 | {success}/{count}（验收{verified}条） | {old[(task, mode)]}/100 |")
            if verified == 100:
                replacements[(task, mode)] = success
                new_baseline[(task, mode)] = success
                key = f"baseline-manifest:{mode}/{task}"
                if key not in memory["recorded"]:
                    memory["recorded"].append(key)
                    with (ROOT / "docs/EXPERIMENT_RESULTS_AND_ANALYSIS.md").open("a") as f:
                        f.write(f"\n\n{now.isoformat()} 官方manifest复测 {task}/{mode}：{success}/100，旧环境{old[(task, mode)]}/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：{directory.parent}。\n")
        memory["history"].append([now.timestamp(), total])
        memory["history"] = memory["history"][-180:]
        deadline = datetime.fromisoformat(state.get("target_deadline_utc", "2026-10-19T01:43:00+00:00"))
        lines += ["", f"目标截止：{deadline.isoformat()}；并行组策略完成{total}/900。"]
        history = [x for x in memory["history"] if now.timestamp() - x[0] <= 1800]
        if len(history) >= 10 and total > history[0][1]:
            rate = (total - history[0][1]) / (now.timestamp() - history[0][0])
            remaining = (900 - total) / rate
            eta = datetime.fromtimestamp(now.timestamp() + remaining, timezone.utc)
            lines.append(f"近30分钟吞吐外推结束时间：{eta.isoformat()}（早期/任务差异会影响估计，不能保证）。")
        if now > deadline and total < 900:
            lines.append("**目标截止已到，尚未完成；保留真实进度并继续运行，不将未完成项计零或宣称已达成。**")
        if state.get("errors"):
            lines += ["", "运行错误："] + [f"- {error}" for error in state["errors"]]
        merged = dict(old)
        merged.update(replacements)
        lines += ["", "## 原模型替换情景", "",
                  "只替换已完整项：传统双鞋Clean92/100，加并行官方manifest预选任务的有效百条结果；未完成沿用旧值。",
                  "这是混合环境/协议的假设汇总，不是新环境50任务全量复现，不按成绩择优。", ""]
        for mode in ["demo_clean", "demo_randomized", "combined"]:
            keys = [k for k in old if mode == "combined" or k[1] == mode]
            before = sum(old[k] for k in keys) / len(keys)
            after = sum(merged[k] for k in keys) / len(keys)
            lines.append(f"- {mode}：{before:.2f}% → {after:.2f}%（{after-before:+.2f}个百分点）。")
        lines += ["", "## Sidecar 20k 新环境", "", "| 任务 | 条件 | 状态 | 成功/完成 |", "|---|---|---|---:|"]
        completed = episodes = 0
        sidecar_complete = {}
        for gpu in [0, 1, 7]:
            queue_state = read(sidecar / f"queue{gpu}/parallel.json", {})
            for job in queue_state.get("jobs", []):
                if job["status"] == "queued":
                    continue
                result = job.get("result")
                count = success = 0
                if result:
                    count, success = result["completed_episodes"], result["successes"]
                elif job.get("output"):
                    path = Path(job["output"]) / job["mode"] / job["task"] / "client.log"
                    if path.exists():
                        with path.open("rb") as f:
                            f.seek(max(0, path.stat().st_size - 131072))
                            log = re.sub(r"\x1b\[[0-9;]*m", "", f.read().decode(errors="replace"))
                        matches = re.findall(r"Success rate:\s*(\d+)/(\d+)", log)
                        if matches:
                            success, count = map(int, matches[-1])
                episodes += count
                complete = job["status"] == "complete" and count == 100
                completed += complete
                if complete:
                    sidecar_complete[(job["task"], job["mode"])] = success
                lines.append(f"| {job['task']} | {job['mode']} | {job['status']} | {success}/{count} |")
                key = f"sidecar-new:{job['mode']}/{job['task']}"
                if complete and key not in memory["recorded"]:
                    memory["recorded"].append(key)
                    with (ROOT / "docs/EXPERIMENT_RESULTS_AND_ANALYSIS.md").open("a") as f:
                        f.write(f"\n\n{now.isoformat()} Sidecar20k新环境 {job['task']}/{job['mode']}：{success}/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。\n")
        lines += ["", f"完整任务条件{completed}/100，策略episode {episodes}/10000；覆盖不足时不报全量宏平均。",
                  "", "加速方法/失败尝试/数值与性能证据：SIDECAR_INFERENCE_ACCELERATION_20261018.md。"]
        common = sorted(set(sidecar_complete) & set(new_baseline))
        lines += ["", "## 当前训练方案的完整任务对照", "",
                  "仅纳入双方均完成100条的共同任务。仿真栈相同，但原版并行manifest固定指令/RNG，Sidecar为传统native；训练数据/表示及编译范围也不同，不能视为逐seed配对或纯结构消融。",
                  "", "| 任务 | 条件 | Sidecar20k | 新环境原版 | 差值（百分点） |", "|---|---|---:|---:|---:|"]
        for task, mode in common:
            s, b = sidecar_complete[(task, mode)], new_baseline[(task, mode)]
            lines.append(f"| {task} | {mode} | {s}% | {b}% | {s-b:+d} |")
        if not common:
            lines.append("尚无双方均完成百条的共同任务；当前不能给训练方案优劣的完整任务结论。")
        elif len(common) >= 3:
            s = sum(sidecar_complete[k] for k in common) / len(common)
            b = sum(new_baseline[k] for k in common) / len(common)
            lines.append(f"共同完整子集共{len(common)}项，描述性宏平均Sidecar {s:.2f}%、原版{b:.2f}%（{s-b:+.2f}个百分点）；这不是全50任务综合平均，也不是结构因果结论。")
        (ROOT / "docs/ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md").write_text("\n".join(lines) + "\n")
        memory_path.write_text(json.dumps(memory, indent=2))
        time.sleep(60)


if __name__ == "__main__":
    main()
