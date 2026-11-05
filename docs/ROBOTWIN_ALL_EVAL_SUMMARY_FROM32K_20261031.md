# RoboTwin evaluation summary from original Sidecar32k

Snapshot: 2026-10-31T05:21:58.792530+00:00. Official source: archived OpenWAM Issue46 table (https://github.com/OpenWAM-Official/OpenWAM/issues/46). One condition is one task × one domain. Whole-suite scope is 50 tasks × Clean/Randomized =100 conditions. Rates are percentages; deltas are percentage points.

## Whole-suite completed evaluations

|Experiment|Checkpoint|Episodes|Clean|Randomized|Overall|Official overall|Delta pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|Original Sidecar|32000|10000|87.66|45.30|66.48|69.04|-2.56|
|Warmup2K (initial version)|24000|3000|87.13|49.27|68.20|69.04|-0.84|
|Full-backward warmup2K|24000|3000|85.80|46.47|66.13|69.04|-2.91|
|Full-backward warmup2K (30 episodes)|32000|3000|89.00|53.20|71.10|69.04|+2.06|
|Full-backward warmup2K (30+70 episodes)|32000|10000|88.02|52.36|70.19|69.04|+1.15|
|Full-backward warmup2K|36000|3000|89.33|51.20|70.27|69.04|+1.23|
|Full-backward warmup2K|42960|3000|89.93|50.07|70.00|69.04|+0.96|
|Heads-only from40000|42960|3000|90.07|50.67|70.37|69.04|+1.33|
|PatchWAM C2R ramp100|20000|3000|84.20|55.33|69.77|69.04|+0.73|
|Warmup2K + full10epoch (1030)|21480|3000|88.67|61.00|74.83|69.04|+5.79|

Official full-suite Clean/Randomized/Overall: 89.36% /48.72% /69.04%.

## Selected subsets and interrupted/running evaluations

|Experiment|Checkpoint|Complete conditions|Episodes counted|Overall|Official same subset|Delta pp|Status|
|---|---:|---:|---:|---:|---:|---:|---|
|Original Sidecar (A100 batch)|42960|20|2000|76.90|78.10|-1.20|stopped|
|Original Sidecar (A100 inference +4090 sim batch)|42960|21|2100|58.90|58.10|+0.81|stopped|
|Reproduced baseline (selected10 tasks)|10740|20|2000|67.70|69.85|-2.15|subset_complete|
|Full-backward warmup2K|20000|35|1050|68.38|72.49|-4.10|interrupted|
|PatchWAM C2R ramp100|24000|56|1680|70.06|66.98|+3.08|stopped_by_user|
|PatchWAM C2R ramp100|16000|4|120|36.67|24.75|+11.92|stopped|
|C2R + warmup2K + joint5epoch (1031)|10740|15|450|69.56|67.73|+1.82|running|

## Interpretation and limits

The highest completed whole-suite rapid result is 1030 warmup2K+full10epoch21480: 74.83%, +5.79pp vs official. Its Clean gap is -0.69pp and Randomized gap +12.28pp. The full-backward32k extended 100-episode result is 70.19%, +1.15pp vs official; its original30 result was71.10%, a0.91pp difference after extending the sample. These are the same checkpoint, not two training experiments.

OriginalSidecar32k and full-backwardwarmup32k are distinct checkpoints/training runs. Original42960 entries are separate stopped execution batches for one checkpoint; they are not two model variants. Heads-only42960 final includes10 L20-sim conditions and90 4090-sim conditions. Full-backward32k100 uses original30+70 audited distinct seeds; it is not claimed equivalent to a single native100 instruction/input sequence. Official reference uses100 episodes/condition; most later experiments use30 and a smaller instruction candidate pool. Language RNG is not frozen across checkpoints. Earlier originalSidecar32k used FFN compilation/cache; later full-backward runs used eager/cache. Preserve these limits when interpreting small deltas or attributing gains to a training change.

Only complete validated conditions count; startup failures and interrupted partial attempts do not enter SR. The1031 current checkpoint has a separate source/training origin from the older reproducedbaseline10740 despite identical step numbers. PatchWAM32000 was cancelled before simulation and has no success rate.

Ancillary hardware comparison: full-backward20k on four matched conditions produced82/120=68.33% on both L20 and4090. This is a small hardware check, not whole-suite evidence. Offline feature/gate diagnostics and throughput experiments are excluded because they have no comparable policy success rate.

## Evidence

- Original Sidecar 32000: `docs/ROBOTWIN_SIDECAR32K_FINAL_RESULTS_20261021.md`
- Warmup2K (initial version) 24000: `docs/ROBOTWIN_WARMUP2K24000_30EP_FINAL_RESULTS_20261024.md`
- Full-backward warmup2K 24000: `outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z/final_results.json`
- Full-backward warmup2K (30 episodes) 32000: `outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/final_results.json`
- Full-backward warmup2K (30+70 episodes) 32000: `outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z/final_results.json`
- Full-backward warmup2K 36000: `outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z/final_results.json`
- Full-backward warmup2K 42960: `outputs/robotwin_eval/full_backward42960_4090_all100_30ep_20261026T010703Z/final_results.json`
- Heads-only from40000 42960: `outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z/final_results.json`
- PatchWAM C2R ramp100 20000: `outputs/robotwin_eval/patchwam_ramp100_20000_20261029/final_results.json`
- Warmup2K + full10epoch (1030) 21480: `outputs/robotwin_eval/warmup2k_full10epoch_21480_4090_all100_30ep_20261030/final_results.json`
- Original Sidecar (A100 batch) 42960: `docs/ROBOTWIN_SIDECAR42960_STOPPED_20261022.md`
- Original Sidecar (A100 inference +4090 sim batch) 42960: `docs/ROBOTWIN_SPLIT42960_STOPPED_20261023.md`
- Reproduced baseline (selected10 tasks) 10740: `docs/ROBOTWIN_BASELINE10740_FINAL_RESULTS_20261022.md`
- Full-backward warmup2K 20000: `outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/final_results.json`
- PatchWAM C2R ramp100 24000: `outputs/robotwin_eval/patchwam_ramp100_24000_20261030/final_results.json`
- PatchWAM C2R ramp100 16000: `docs/ROBOTWIN_PATCHWAM_RAMP100_16000_LIVE_20261029.md`
- C2R + warmup2K + joint5epoch (1031) 10740: `outputs/robotwin_eval/warmup2k_fromc2r_5epoch_10740_4090_all100_30ep_20261031/comparison.json`
