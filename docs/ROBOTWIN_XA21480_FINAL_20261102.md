# XA 10epoch最终RoboTwin速测

100条件×30条＝3000条，2178成功，最终72.60%；Clean1335/1500=89.00%，Randomized843/1500=56.20%。原21480重载10epoch为74.83%/88.67%/61.00%，XA综合-2.23pp、Clean+0.33pp、Randomized-4.80pp。官方69.04%/89.36%/48.72%，XA综合+3.56pp。

独立验收通过：outputs/robotwin_eval/xa21480_4090_all100_30ep_20261101/final_independent_audit.json，精确XA SHA f55e7641237f28bb14f004237b2cafaaad8dd7fc55d9c9d43537340075ede1d7。失败partial不计SR。相同eager/cache .99/max3、10 denoise、horizon32、no video、RT/原生预算/专家筛seed。

XA为训练阶段action hidden alignment，评估保留完整Sidecar，使用双卡放置T5。2026-11-02新增同XA权重单卡与既有split的2固定fixture直接对照，动作逐元素一致/maxabs0；其余实际在线输入仍由一条仿真逐chunk镜像检查，不将固定2输入结论外推所有状态。XA训练增强、loss和起点等需结合训练配置分析，不能单用本组SR证明alignment单独的普遍作用。

后续分析已由用户授权：原最佳21480的Full/Write-Off/Relation-Reset，10典型任务、双域、同manifest、每条件10条的600条配对初筛。见docs/ROBOTWIN_21480_INTERVENTIONS_SETUP_20261102.md。


2026-11-03 03:30 UTC补测修正：原move_stapler_pad/demo_randomized日志含OIDN Error的13/30不作为最终有效成绩，独立新run refineoff21480_all50_30ep_20261103中xa_repair完整30条为11/30，native exit0、完整30条和无渲染错误检查通过。旧原始记录保留，修正总计2176/3000=72.5333%，Clean1335/1500=89%，Randomized841/1500=56.0667%。修正汇总见新run/comparison.json的xa_repair，原目录final_results未覆盖。
