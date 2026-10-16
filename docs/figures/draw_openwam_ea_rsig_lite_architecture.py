"""Draw the audited Lite C-joint architecture; document artifact only.

SVG needs only Python's standard library. PNG rendering optionally uses resvg-py
and --font-file, installed in a separate drawing environment, never training .venv.
"""
from pathlib import Path
from html import escape
import argparse

P = {"video": ("#eaf2ff", "#306bd8"), "action": ("#f1ebff", "#7746ce"),
     "ea": ("#e8f8ed", "#168044"), "shared": ("#f0f2f6", "#778394"),
     "loss": ("#fff0f2", "#c83c57"), "white": ("#ffffff", "#a5b0bd")}
items, edges, boxes = [], [], []


def text(x, y, value, size=32, color="#15253e", anchor="middle", weight=400):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def node(x, y, w, h, lines, kind="shared", size=32):
    fill, stroke = P[kind]
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>']
    step = size * 1.48
    start = y + h / 2 - (len(lines) - 1) * step / 2 + size * .34
    s.extend(text(x+w/2, start+i*step, line, size, weight=600 if i == 0 else 400)
             for i, line in enumerate(lines))
    boxes.append("".join(s))


def edge(points, kind="shared", dashed=False, arrow=True, width=5):
    color = P[kind][1]
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    # White underlay makes crossings visibly distinct from junctions.
    edges.append(f'<path d="{d}" fill="none" stroke="white" stroke-width="{width+5}"/>')
    attrs = ' stroke-dasharray="13 10"' if dashed else ""
    if arrow:
        attrs += f' marker-end="url(#{kind})"'
    edges.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linejoin="round"{attrs}/>')


def panel(x, y, w, h, label, fill):
    items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" '
                 f'fill="{fill}" stroke="#c8d2df" stroke-width="3"/>')
    items.append(text(x+30, y+60, label, 42, anchor="start", weight=700))


def frame(x, y, w, h, kind):
    fill, stroke = P[kind]
    items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>')


def build():
    panel(40, 200, 2150, 1960, "A  OpenWAM 主干 + EA 的两个接点（从上往下读）", "#fbfcff")
    panel(2230, 200, 1330, 1960, "B  EA 内部结构（绿色为新增）", "#f6fcf7")
    panel(40, 2190, 3520, 350, "C  训练监督：标签进入 loss，不进入前向网络", "#fffafb")
    items.extend([text(80, 100, "OpenWAM-alpha + EA-RSIG-Lite", 72, anchor="start", weight=700),
                  text(85, 161, "已实现的 C joint 结构  ·  Foundation 初始化  ·  保留原视频 / 动作联合注意力", 32, anchor="start")])
    for x, kind, label in [(2250,"video","原 Video"),(2600,"action","原 Action"),(2960,"ea","新增 EA")]:
        fill, stroke = P[kind]
        items.append(f'<rect x="{x}" y="73" width="52" height="32" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        items.append(text(x+68, 103, label, 31, anchor="start"))
    items.append(text(2250, 160, "实线：前向数据流    红虚线：仅训练监督", 30, anchor="start"))

    # Main model: video and action inputs, current-state condition.
    node(120,400,530,125,["三视角视频", "33帧跨度 → stride 4 → 9帧"],"video",30)
    node(120,580,530,110,["Wan VAE 编码（冻结）"],"video")
    node(120,780,530,110,["未来 latent 加噪 τv", "当前首帧保持干净"],"video",30)
    node(120,950,530,95,["Video patch embedding"],"video",30)
    node(1450,400,550,125,["未来32步绝对 EEF 动作", "20D有效维度 → 80D统一空间"],"action",30)
    node(1450,780,550,110,["动作加噪 τa"],"action")
    node(1450,950,550,95,["Action Encoder"],"action")
    node(1450,580,550,110,["任务文本 + 当前 Proprio", "原 OpenWAM 条件输入"],"shared",30)
    node(770,940,550,115,["Condition", "送入联合块两侧 Cross-Attn"],"shared",29)
    edge([(385,525),(385,580)],"video")
    edge([(385,690),(385,780)],"video")
    edge([(385,890),(385,950)],"video")
    edge([(1725,525),(1725,550),(2055,550),(2055,735),(1725,735),(1725,780)],"action")
    edge([(1725,890),(1725,950)],"action")
    edge([(1450,635),(1370,635),(1370,995),(1320,995)])

    # The only EA feature source, selected before all Video DiT blocks.
    node(780,570,400,120,["① 仅当前首帧", "干净 latent z_t"],"ea",31)
    node(780,760,400,115,["共享 patch embedding", "不额外跑 Video DiT"],"ea",28)
    edge([(650,635),(780,635)],"ea")
    edge([(980,690),(980,760)],"ea")
    edge([(1180,817),(1235,817),(1235,320),(2575,320),(2575,400)],"ea",width=7)
    items.append(text(1740,309,"当前干净 patch → EA",29,color=P["ea"][1],weight=600))

    # First original mixed block, followed by a single Action-only EA adapter.
    frame(120,1120,1880,280,"shared")
    boxes.append(text(1060,1168,"第1个联合块 · block 0",35,weight=700))
    node(170,1190,650,55,["Video：AdaLN → QKV"],"video",28)
    node(1300,1190,650,55,["Action：AdaLN → QKV"],"action",28)
    node(170,1265,1780,60,["Shared Video–Action Attention（原 mutual attention）"],"shared",30)
    node(170,1340,650,45,["Cross-Attn → AdaLN → FFN"],"video",26)
    node(1300,1340,650,45,["Cross-Attn → AdaLN → FFN"],"action",26)
    edge([(385,1045),(385,1120)],"video")
    edge([(1725,1045),(1725,1120)],"action")
    edge([(1045,1055),(1045,1120)])
    edge([(495,1400),(495,1790)],"video",width=7)

    frame(1325,1470,675,220,"ea")
    boxes.append(text(1345,1505,"② 新增 Action 条件残差",29,anchor="start",weight=700))
    node(1350,1525,235,55,["Q：Action hidden"],"action",25)
    node(1350,1605,235,55,["K,V：EA tokens"],"ea",25)
    node(1635,1525,320,75,["8-head Cross-Attn"],"ea",27)
    node(1635,1620,320,55,["Linear（零初始化）"],"ea",26)
    edge([(1625,1400),(1625,1440),(1468,1440),(1468,1525)],"action")
    edge([(1585,1553),(1635,1553)],"action")
    edge([(1585,1633),(1608,1633),(1608,1580),(1635,1580)],"ea")
    edge([(1795,1600),(1795,1620)],"ea")
    edge([(1795,1675),(1795,1695),(1625,1695),(1625,1705)],"ea")
    edge([(1625,1400),(2120,1400),(2120,1730),(1625,1730)],"action",width=6)
    boxes.append('<circle cx="1625" cy="1730" r="25" fill="white" stroke="#168044" stroke-width="4"/>')
    boxes.append(text(1625,1742,"+",38,weight=700))
    boxes.append(text(2010,1780,"identity",25,color=P["action"][1]))
    edge([(1625,1755),(1625,1790)],"action",width=7)
    node(120,1790,1880,160,["后续联合块 · blocks 1 … L−1", "继续原视频 / 动作共享注意力与各自 FFN"],"shared",34)
    node(180,2020,630,90,["Video head：latent 流速度"],"video",31)
    node(1320,2020,630,90,["Action head：动作流速度"],"action",31)
    edge([(495,1950),(495,2020)],"video")
    edge([(1625,1950),(1625,2020)],"action")
    items.append(text(1080,1993,"原联合序列、attention mask、RoPE 均保持；没有直接新增 Video 残差",27))
    items.append(text(495,2147,"监督：L_video",27,color=P["loss"][1]))
    items.append(text(1635,2147,"监督：L_action",27,color=P["loss"][1]))

    # Expanded EA module. Every type is preserved in the action-token assembly.
    node(2290,400,570,130,["当前三视角 patch", "每视角14×14 → 共588 tokens"],"video",30)
    node(2910,400,580,130,["任务文本", "Masked mean → Linear"],"shared",30)
    node(2290,600,570,125,["LayerNorm + Linear → 512D", "文本通道调制（保持空间网格）"],"ea",28)
    node(2910,610,580,115,["当前双臂 TCP [B,2,10]", "+ arm mask [B,2]"],"shared",30)
    node(2910,805,580,110,["EEF MLP：10 → 512 → 512"],"ea",28)
    edge([(2575,530),(2575,600)],"ea")
    edge([(2910,465),(2885,465),(2885,662),(2860,662)],"ea")
    edge([(3200,725),(3200,805)],"ea")
    node(2290,835,570,220,["Ground · 空间注意力", "4个 Object / Goal learned query", "QK^T → softmax → 加权汇聚", "4个语义 slot + 存在性门控"],"ea",28)
    edge([(2575,725),(2575,835)],"ea")
    node(2290,1160,1200,160,["Relation · 关系 MLP", "1536 → 1024 → 512 + LayerNorm", "2个 EEF–Object + 2个 Object–Goal 关系 token"],"ea",30)
    edge([(2575,1055),(2575,1160)],"ea")
    edge([(3200,915),(3200,1100),(3050,1100),(3050,1160)],"ea")
    node(2290,1430,570,180,["Progress · 注意力汇聚", "1个 query + 8-head Attn + FFN", "1个512D进展 token"],"ea",28)
    node(2910,1430,580,180,["Plan · 多时间尺度 MLP", "8 / 16 / 32步：方向 + 运动概率", "6个 token（3时刻 × 2臂）"],"ea",28)
    edge([(2575,1320),(2575,1430)],"ea")
    edge([(2860,1520),(2910,1520)],"ea")
    edge([(3490,860),(3520,860),(3520,1520),(3490,1520)],"ea")
    edge([(2860,945),(3520,945)],"ea",arrow=False)
    items.append(text(3310,1408,"EEF / Object / Goal",24,color=P["ea"][1]))
    node(2290,1770,1200,190,["15 × 512 交互 token", "4 Ground + 4 Relation + 1 Progress + 6 Plan", "各组投影 + 预测有效性门控"],"ea",33)
    edge([(2575,1610),(2575,1770)],"ea")
    edge([(3200,1610),(3200,1770)],"ea")
    edge([(2290,945),(2255,945),(2255,1865),(2290,1865)],"ea")
    edge([(2290,1240),(2255,1240)],"ea",arrow=False)
    # Sole model injection: an external green bus entering the adapter's KV port.
    edge([(2290,1865),(2208,1865),(2208,1450),(1250,1450),(1250,1633),(1350,1633)],"ea",width=7)
    items.append(text(1900,1438,"15个 EA tokens → K/V",26,color=P["ea"][1],weight=600))
    node(2290,2010,1200,110,["并行辅助输出：空间定位 / 接触与接近 / 运动方向与概率", "从 Ground / Relation / Plan 中间特征预测，仅用于监督"],"ea",28)

    # Supervision is a distinct, training-only branch.
    node(100,2300,920,160,["原始 EA oracle → 离线重建8/16/32标签", "labels/*.npz：130D监督目标", "物体/目标/接触等真值仅用于监督"],"loss",29)
    node(1110,2300,700,160,["EA 辅助损失", "L_ground / L_relation / L_plan", "不作为 policy 输入"],"loss",30)
    edge([(1020,2380),(1110,2380)],"loss",dashed=True)
    edge([(2890,2120),(2890,2170),(1460,2170),(1460,2300)],"loss",dashed=True)
    node(1900,2300,1580,160,["L = L_video + L_action", "+ 0.025 L_ground + 0.01 L_relation + 0.05 L_plan", "joint更新：Video DiT + Action DiT + EA；VAE/文本编码器冻结"],"white",30)
    items.append(text(1800,2514,"推理只用当前 RGB + 文本 + 当前 EEF/TCP；Baseline 对照关闭绿色 EA 分支，其余训练协议对齐。",31,weight=600))
    defs = ''.join(f'<marker id="{k}" markerWidth="12" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5 L0,10 Z" fill="{v[1]}"/></marker>' for k,v in P.items())
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="3600" height="2600" viewBox="0 0 3600 2600">'
            '<title>OpenWAM-alpha + EA-RSIG-Lite C joint architecture</title>'
            '<desc>Explicit current-frame EA branch, relation/plan modules, Action-only block0 residual, and isolated privileged supervision.</desc>'
            f'<defs>{defs}</defs><rect width="3600" height="2600" fill="white"/>'
            '<g font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif">'
            + ''.join(items+edges+boxes) + '</g></svg>')


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-file", type=Path)
    args=parser.parse_args()
    stem=Path(__file__).parent / "OpenWAM_alpha_EA_RSIG_Lite_architecture_20261016"
    svg=build();stem.with_suffix('.svg').write_text(svg)
    if args.font_file:
        import resvg_py
        stem.with_suffix('.png').write_bytes(resvg_py.svg_to_bytes(svg_string=svg,
            font_files=[str(args.font_file)], skip_system_fonts=True, background="white"))
    print(stem)
