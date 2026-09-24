"""SRT 字幕工具后端：时间校准 / 片段剪辑 / 双语合成。

校准：根据用户给出的若干「原字幕时刻 -> 视频实际时刻」锚点，
按原时刻排序后在相邻锚点间做线性插值，两端按最近一段的斜率外推，
统一换算每条字幕的起止时间，并检测负时间 / 起止倒置 / 与下一条重叠。

剪辑：按左闭右开保留区间裁剪字幕并拼接到连续时间轴。

双语合成：主轨与译文轨按时间顺序全局配对（每条主轨 0~2 条译文、
译文至多使用一次、仅正时长交叠可配），导出按主轨时间编号的双语 SRT。
"""
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI(title="SRT 字幕时间校准 / 片段剪辑 / 双语合成")

STATIC_DIR = Path(__file__).parent / "static"

TIME_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*$")
ARROW_RE = re.compile(r"-->")


# ---------------------------------------------------------------- 时间工具

def parse_time(s: str) -> int:
    """把 'HH:MM:SS,mmm' 解析为毫秒整数，失败抛 ValueError。"""
    m = TIME_RE.match(s or "")
    if not m:
        raise ValueError(f"无法解析时间: {s!r}（应为 HH:MM:SS,mmm）")
    h, mnt, sec, ms = (int(g) for g in m.groups())
    ms = int(str(m.group(4)).ljust(3, "0"))  # 允许 1~3 位毫秒
    return ((h * 60 + mnt) * 60 + sec) * 1000 + ms


def fmt_time(ms: int) -> str:
    """毫秒格式化为 SRT 时间；负数带负号以便预览中识别。"""
    sign = "-" if ms < 0 else ""
    ms = abs(int(round(ms)))
    h, ms = divmod(ms, 3600_000)
    mnt, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{mnt:02d}:{sec:02d},{ms:03d}"


# ---------------------------------------------------------------- SRT 解析

def parse_srt(text: str) -> List[dict]:
    """解析 SRT，保留序号与正文。返回 [{index, start, end, lines}]。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n﻿ ")
    if not text:
        raise ValueError("SRT 内容为空")
    entries = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.strip("\n").split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # 找到包含 --> 的时间行，其上一行（若有）为序号
        ti = next((i for i, ln in enumerate(lines) if ARROW_RE.search(ln)), None)
        if ti is None:
            raise ValueError(f"缺少时间行 '-->'：{lines[0][:40]!r}")
        parts = ARROW_RE.split(lines[ti])
        if len(parts) != 2:
            raise ValueError(f"时间行格式错误：{lines[ti]!r}")
        # 去掉时间行中可能附带的定位信息（如 'X1:.. X2:..'）
        start_s = parts[0].strip()
        end_s = parts[1].strip().split()[0] if parts[1].strip() else ""
        try:
            start, end = parse_time(start_s), parse_time(end_s)
        except ValueError as e:
            raise ValueError(f"第 {len(entries) + 1} 块：{e}")
        index = lines[0].strip() if ti > 0 else str(len(entries) + 1)
        entries.append({
            "index": index,
            "start": start,
            "end": end,
            "lines": lines[ti + 1:],
        })
    if not entries:
        raise ValueError("未解析到任何字幕块")
    return entries


# ---------------------------------------------------------------- 锚点与映射

def build_mapper(anchors_ms: List[tuple]):
    """anchors_ms: [(orig_ms, actual_ms)]，已排序且已校验。
    返回分段线性映射函数：相邻锚点线性插值，两端按最近一段斜率外推。"""
    pts = anchors_ms

    def map_t(t: float) -> float:
        if t <= pts[0][0]:
            (o0, a0), (o1, a1) = pts[0], pts[1]
        elif t >= pts[-1][0]:
            (o0, a0), (o1, a1) = pts[-2], pts[-1]
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= t <= pts[i + 1][0]:
                    (o0, a0), (o1, a1) = pts[i], pts[i + 1]
                    break
        slope = (a1 - a0) / (o1 - o0)
        return a0 + (t - o0) * slope

    return map_t


def validate_anchors(raw: List["Anchor"]) -> List[tuple]:
    """校验并排序锚点，返回 [(orig_ms, actual_ms)]。失败抛 ValueError。"""
    if len(raw) < 2:
        raise ValueError("至少需要 2 个锚点")
    pts = []
    for i, a in enumerate(raw, 1):
        try:
            pts.append((parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            raise ValueError(f"锚点 {i}：{e}")
    pts.sort(key=lambda p: p[0])
    for i in range(1, len(pts)):
        if pts[i][0] == pts[i - 1][0]:
            raise ValueError(f"锚点原时刻重复：{fmt_time(pts[i][0])}")
        if pts[i][1] <= pts[i - 1][1]:
            raise ValueError(
                f"映射后的实际时刻必须严格递增：{fmt_time(pts[i - 1][1])} 之后不能是 {fmt_time(pts[i][1])}"
            )
    return pts


def plan_anchor_suggestion(raw: List["Anchor"]) -> dict:
    """锚点校验失败时的异常定位：求需要剔除锚点最少的修复方案。

    按原时刻排序后，选出「原时刻、实际时刻都严格递增」的最长子序列（至少 2 点）。
    多解时决胜顺序：
      1) 最长（剔除行数最少）；
      2) 首尾原时刻（全局最小/最大原时刻）锚点的保留数量最多；
      3) 剔除项的原输入行号序列字典序最小。
    返回规划字典；无法凑出 2 点时 feasible=False（无法预演）。
    """
    n = len(raw)
    if n < 2:
        return {"feasible": False, "reason": "至少需要 2 个锚点，无法预演；请补充锚点后重试。"}

    parsed = []  # (输入行号, 原时刻ms, 实际时刻ms)
    bad = {}     # 无法解析的行号 -> 原因（必然要剔除）
    for i, a in enumerate(raw, 1):
        try:
            parsed.append((i, parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            bad[i] = str(e)
    if len(parsed) < 2:
        return {
            "feasible": False,
            "reason": "时间格式可解析的有效锚点不足 2 个，无法预演；请按 HH:MM:SS,mmm 修正时间格式。",
        }

    # 按原时刻排序（相同原时刻再按实际时刻、行号兜底，保证确定性）；
    # 转移时同时要求 orig、actual 严格小于，故重复原时刻不会串链。
    nodes = sorted(parsed, key=lambda p: (p[1], p[2], p[0]))
    m = len(nodes)
    row_orig = {r: o for r, o, _ in parsed}
    row_actual = {r: c for r, _, c in parsed}
    min_o = min(o for _, o, _ in parsed)
    max_o = max(o for _, o, _ in parsed)
    all_rows = set(range(1, n + 1))
    orig_count: dict = {}
    for _, o, _ in parsed:
        orig_count[o] = orig_count.get(o, 0) + 1

    def endpoint_hits(chain: tuple) -> int:
        """链中属于首/尾原时刻（全局最小、最大原时刻）的锚点数。"""
        return sum(1 for r in chain if row_orig[r] == min_o or row_orig[r] == max_o)

    def better(c1: tuple, c2: tuple) -> bool:
        """完整保留链的优劣比较：长度 > 首尾保留数 > 剔除行号序列字典序。"""
        if len(c1) != len(c2):
            return len(c1) > len(c2)
        e1, e2 = endpoint_hits(c1), endpoint_hits(c2)
        if e1 != e2:
            return e1 > e2
        rem1 = tuple(sorted(all_rows.difference(c1)))
        rem2 = tuple(sorted(all_rows.difference(c2)))
        return rem1 < rem2

    # O(m^2) DP：dp[i] 为以 nodes[i] 结尾的最优保留链（链元素为输入行号）
    dp = []
    for i in range(m):
        ri, oi, ai = nodes[i]
        best = (ri,)
        for j in range(i):
            rj, oj, aj = nodes[j]
            if oj < oi and aj < ai:
                cand = (*dp[j], ri)
                if better(cand, best):
                    best = cand
        dp.append(best)
    chain = dp[0]
    for c in dp[1:]:
        if better(c, chain):
            chain = c

    if len(chain) < 2:
        return {
            "feasible": False,
            "reason": "找不到两个原时刻与实际时刻都严格递增的锚点，无法预演；请手动修正锚点时刻。",
        }

    kept_rows = set(chain)
    removed = []
    for r in sorted(all_rows - kept_rows):
        if r in bad:
            reason = bad[r]
        elif orig_count[row_orig[r]] > 1:
            reason = f"原时刻 {fmt_time(row_orig[r])} 与其他锚点重复"
        else:
            reason = "实际时刻未随原时刻严格递增，剔除后保留锚点整体最优"
        removed.append({
            "row": r,
            "orig": raw[r - 1].orig,
            "actual": raw[r - 1].actual,
            "reason": reason,
        })
    kept = [
        {"row": r, "orig": fmt_time(row_orig[r]), "actual": fmt_time(row_actual[r])}
        for r in chain
    ]
    pts = [(row_orig[r], row_actual[r]) for r in chain]
    return {
        "feasible": True,
        "remove_rows": sorted(all_rows - kept_rows),
        "kept": kept,
        "removed": removed,
        "_pts": pts,
    }


def calibrate(entries: List[dict], map_t) -> List[dict]:
    """换算每条字幕起止时间并打问题标记。"""
    out = []
    for e in entries:
        ns, ne = int(round(map_t(e["start"]))), int(round(map_t(e["end"])))
        flags = []
        if ns < 0 or ne < 0:
            flags.append("负时间")
        if ne <= ns:
            flags.append("起止倒置")
        out.append({**e, "new_start": ns, "new_end": ne, "flags": flags})
    for i in range(len(out) - 1):
        if out[i]["new_end"] > out[i + 1]["new_start"]:
            out[i]["flags"].append("与下一条重叠")
    return out


# ---------------------------------------------------------------- API

class Anchor(BaseModel):
    orig: str    # 原字幕时刻 HH:MM:SS,mmm
    actual: str  # 视频实际时刻 HH:MM:SS,mmm


class CalibrateRequest(BaseModel):
    srt: str
    anchors: List[Anchor]


def _prepare(req: CalibrateRequest):
    """公共预处理，成功返回 (entries, calibrated)，失败抛 400。

    锚点校验类错误 detail 为 {"type": "anchor_invalid", "message": ...}，
    便于前端据此调用 /api/suggest 获取剔除建议；SRT 解析错误仍返回纯文本。
    """
    try:
        pts = validate_anchors(req.anchors)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail={"type": "anchor_invalid", "message": str(e)}
        )
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return calibrate(entries, build_mapper(pts))


def _preview_payload(rows: List[dict]) -> dict:
    return {
        "ok": all(not r["flags"] for r in rows),
        "problem_count": sum(1 for r in rows if r["flags"]),
        "entries": [
            {
                "index": r["index"],
                "old": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "new": f"{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}",
                "flags": r["flags"],
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/preview")
def preview(req: CalibrateRequest):
    rows = _prepare(req)
    return _preview_payload(rows)


@app.post("/api/suggest")
def suggest(req: CalibrateRequest):
    """锚点校验失败时的异常定位建议。

    返回需要剔除的最少锚点；确认前不改动任何输入，确认后由前端按原流程
    （/api/preview、/api/export）继续。无法凑出 2 个有效锚点时提示无法预演。
    """
    plan = plan_anchor_suggestion(req.anchors)
    if not plan["feasible"]:
        return {"feasible": False, "reason": plan["reason"]}

    pts = plan.pop("_pts")
    # 用保留锚点复用现有校准预览，展示受影响字幕；SRT 本身无法解析时仅提示
    try:
        rows = calibrate(parse_srt(req.srt), build_mapper(pts))
        plan["preview"] = _preview_payload(rows)
        plan["preview_error"] = None
    except ValueError as e:
        plan["preview"] = None
        plan["preview_error"] = str(e)
    return plan


# ---------------------------------------------------------------- 剪辑片段

class KeepInterval(BaseModel):
    start: str  # 原视频保留区间起点 HH:MM:SS,mmm
    end: str    # 原视频保留区间终点 HH:MM:SS,mmm


class ClipRequest(BaseModel):
    srt: str
    intervals: List[KeepInterval]


def validate_intervals(raw: List["KeepInterval"]) -> List[tuple]:
    """校验保留区间并按起点排序，返回 [(start_ms, end_ms, 输入行号)]。

    规则：起点须严格早于终点；区间为左闭右开，故相邻区间端点相触
    （前段终点 == 后段起点）不算重叠，严格交叉才报错。
    """
    rows = [r for r in raw if r.start.strip() or r.end.strip()]
    if not rows:
        raise ValueError("请至少填写 1 个保留区间")
    parsed = []
    for i, r in enumerate(rows, 1):
        try:
            s, e = parse_time(r.start), parse_time(r.end)
        except ValueError as ex:
            raise ValueError(f"保留区间 {i}：{ex}")
        if s >= e:
            raise ValueError(
                f"保留区间 {i}：起点 {fmt_time(s)} 须早于终点 {fmt_time(e)}"
            )
        parsed.append((s, e, i))
    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    # 按起点排序后，相邻区间若不重叠则任意区间都不可能重叠
    for k in range(1, len(parsed)):
        ps, pe, prow = parsed[k - 1]
        s, e, row = parsed[k]
        if s < pe:  # 左闭右开：s == pe 只是端点相触，允许
            raise ValueError(
                f"保留区间重叠：第 {prow} 行（{fmt_time(ps)} – {fmt_time(pe)}）"
                f"与第 {row} 行（{fmt_time(s)} – {fmt_time(e)}）"
            )
    return parsed


def clip_entries(entries: List[dict], intervals: List[tuple]):
    """按左闭右开区间裁剪字幕，返回 (新字幕列表, 保留区间总时长ms)。

    对每条字幕 [start, end) 与每个保留区间求交集：空交集丢弃；跨多个
    区间的字幕按区间拆成多条，正文原样保留。交集时间换算到拼接后的
    连续时间轴：新时间 = 交集时间 - 区间起点 + 此前所有区间时长之和。
    输出按新起点排序（起点相同保持原顺序）并从 1 连续重编号。
    """
    out = []
    offset = 0
    for s, e, _row in intervals:
        for order, entry in enumerate(entries):
            lo, hi = max(entry["start"], s), min(entry["end"], e)
            if lo < hi:  # 左闭右开：lo == hi 为空交集
                out.append({
                    "index": "",
                    "src_index": entry["index"],
                    "src_order": order,
                    "old_start": lo,
                    "old_end": hi,
                    "start": offset + lo - s,
                    "end": offset + hi - s,
                    "lines": entry["lines"],
                })
        offset += e - s
    out.sort(key=lambda r: (r["start"], r["src_order"]))
    for n, r in enumerate(out, 1):
        r["index"] = str(n)
    return out, offset


def _clip_prepare(req: ClipRequest):
    """剪辑功能公共预处理，成功返回 (entries, intervals)，失败抛 400。"""
    try:
        intervals = validate_intervals(req.intervals)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, intervals


@app.post("/api/clip/preview")
def clip_preview(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, total_ms = clip_entries(entries, intervals)
    return {
        "entry_count": len(rows),
        "interval_count": len(intervals),
        "total_duration": fmt_time(total_ms),
        "entries": [
            {
                "index": r["index"],
                "source": r["src_index"],
                "old": f"{fmt_time(r['old_start'])} --> {fmt_time(r['old_end'])}",
                "new": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.post("/api/clip/export")
def clip_export(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, _ = clip_entries(entries, intervals)
    if not rows:
        raise HTTPException(
            status_code=400, detail="所有字幕均与保留区间无交集，没有可导出的内容"
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="clipped.srt"'},
    )


@app.post("/api/export")
def export(req: CalibrateRequest):
    rows = _prepare(req)
    bad = [r for r in rows if r["flags"]]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"存在 {len(bad)} 条问题字幕（负时间/起止倒置/重叠），已禁止导出",
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="calibrated.srt"'},
    )


# ---------------------------------------------------------------- 双语 SRT 合成

class BilingualRequest(BaseModel):
    main: str   # 主轨 SRT 文本（决定导出时间与编号）
    trans: str  # 译文轨 SRT 文本


def timed_entries(text: str, label: str) -> List[dict]:
    """解析一轨 SRT，校验正时长并按时间排序；每条附加 pos（原文件顺序，0 起）。

    空时长（终点 == 起点）或起止倒置（终点 < 起点）均报错。
    """
    try:
        entries = parse_srt(text)
    except ValueError as e:
        raise ValueError(f"{label}{e}")
    for k, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            raise ValueError(
                f"{label}第 {k} 块（序号 {e['index']}）时长为空或起止倒置："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )
    for pos, e in enumerate(entries):
        e["pos"] = pos
    entries.sort(key=lambda e: (e["start"], e["end"], e["pos"]))
    return entries


def pair_bilingual(mains: List[dict], trans: List[dict]):
    """按时间顺序全局配对：每条主轨配 0~2 条译文，每条译文至多使用一次，
    仅存在正时长交叠的主/译文才可配对；配对保持时间顺序（先出现的主轨
    配到的译文序号都小于后出现的主轨所配译文）。目标依次：
      1) 总交叠毫秒数最大；
      2) 漏配译文最少（已配译文条数最多）；
      3) 已配译文的原序号序列字典序最小（等价于让译文使用位图尽量靠前置位，
         即用位值 bit[j] = 1 << (n-1-j) 表示译文 j 被使用，位值总和最大者胜）。
    返回 (pairs, best)：pairs 为 [(主轨下标, (译文下标, ...))]，best 为目标键。

    实现：只在「译文 j 与主轨 i 正交叠」的决策点分支，强制步骤（丢弃对当前
    及后续主轨均无交叠的译文、收尾与当前及后续译文均无交叠的主轨）直接快进，
    状态数与交叠对数同阶；所有边都使 i+j 严格增大，构成 DAG。
    """
    m, n = len(mains), len(trans)
    MS = [e["start"] for e in mains]
    ME = [e["end"] for e in mains]
    TS = [e["start"] for e in trans]
    TE = [e["end"] for e in trans]
    bit = [1 << (n - 1 - j) for j in range(n)]

    def ff(i: int, j: int, k: int):
        """快进强制步骤，返回下一个决策点（或 i==m / j==n 的终点）。"""
        while i < m and j < n:
            if TE[j] <= MS[i]:      # 译文 j 对当前及后续主轨都无交叠：丢弃
                j += 1
            elif TS[j] >= ME[i]:    # 当前及后续译文都与主轨 i 无交叠：主轨 i 收尾
                i += 1
                k = 0
            else:
                break
        return i, j, k

    def moves(i: int, j: int, k: int):
        """决策点 (i, j, k)：译文 j 与主轨 i 正交叠，主轨 i 已配 k 条。
        产出 (交叠增益, 用译文数, 位值增益, ni, nj, nk, 本步配出的译文或 None)。"""
        if i >= m or j >= n:
            return
        if k < 2:  # 主轨 i 配译文 j
            w = min(ME[i], TE[j]) - max(MS[i], TS[j])
            yield (w, 1, bit[j], *ff(i + (k == 1), j + 1, 0 if k == 1 else 1), j)
        yield (0, 0, 0, *ff(i, j + 1, k), None)      # 译文 j 留作未配
        yield (0, 0, 0, *ff(i + 1, j, 0), None)      # 主轨 i 收尾，译文 j 留给后续主轨

    # 收集可达状态
    MAX_STATES = 200_000
    start = ff(0, 0, 0)
    states = {start}
    stack = [start]
    while stack:
        for *_g, ni, nj, nk, _u in moves(*stack.pop()):
            t = (ni, nj, nk)
            if t not in states:
                states.add(t)
                stack.append(t)
        if len(states) > MAX_STATES:
            raise ValueError("配对状态规模过大，请检查两轨是否存在大量互相覆盖的超长字幕")

    # 自后向前松弛：suff[s] = 从 s 到终点的最优目标键 (总交叠, 已配译文数, 位值)
    suff = {}
    for s in sorted(states, key=lambda p: p[0] + p[1], reverse=True):
        best = None
        for g0, g1, g2, ni, nj, nk, _u in moves(*s):
            t = suff[(ni, nj, nk)]
            key = (g0 + t[0], g1 + t[1], g2 | t[2])
            if best is None or key > best:
                best = key
        suff[s] = best if best is not None else (0, 0, 0)

    # 自前向后还原：任取达到最优键的转移即可（位值唯一决定已配译文集合）
    pairs = []
    i, j, k = start
    cur: List[int] = []
    while i < m and j < n:
        for g0, g1, g2, ni, nj, nk, used in moves(i, j, k):
            t = suff[(ni, nj, nk)]
            if (g0 + t[0], g1 + t[1], g2 | t[2]) == suff[(i, j, k)]:
                if used is not None:
                    cur.append(used)
                if ni != i:  # 主轨 i 收尾
                    if cur:
                        pairs.append((i, tuple(cur)))
                    cur = []
                i, j, k = ni, nj, nk
                break
    if cur:  # 译文耗尽时主轨 i 可能刚配到 1 条，一并收尾
        pairs.append((i, tuple(cur)))
    return pairs, suff[start]


def _bilingual_prepare(req: "BilingualRequest"):
    """双语合成公共预处理，成功返回 (mains, trans, plan)，失败抛 400。

    plan 为按主轨时间顺序的 [(主轨下标, (译文下标, ...))]，覆盖全部主轨
    （未配到译文的主轨对应空元组）。
    """
    try:
        mains = timed_entries(req.main, "主轨：")
        trans = timed_entries(req.trans, "译文轨：")
        pairs, _best = pair_bilingual(mains, trans)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    by_main = {i: used for i, used in pairs}
    plan = [(i, by_main.get(i, ())) for i in range(len(mains))]
    return mains, trans, plan


def _entry_payload(e: dict) -> dict:
    return {
        "index": e["index"],
        "time": f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}",
        "text": "\n".join(e["lines"]),
    }


@app.post("/api/bilingual/preview")
def bilingual_preview(req: BilingualRequest):
    mains, trans, plan = _bilingual_prepare(req)
    used_set = {j for _i, used in plan for j in used}
    entries = []
    for i, used in plan:
        e = mains[i]
        translations = []
        for j in used:
            t = trans[j]
            ov = min(e["end"], t["end"]) - max(e["start"], t["start"])
            translations.append({**_entry_payload(t), "overlap": ov})
        entries.append({**_entry_payload(e), "translations": translations})
    return {
        "main_count": len(mains),
        "trans_count": len(trans),
        "paired_trans_count": len(used_set),
        "unpaired_trans_count": len(trans) - len(used_set),
        "total_overlap": sum(
            min(mains[i]["end"], trans[j]["end"]) - max(mains[i]["start"], trans[j]["start"])
            for i, used in plan for j in used
        ),
        "entries": entries,
        "unpaired": [
            _entry_payload(t) for j, t in enumerate(trans) if j not in used_set
        ],
    }


@app.post("/api/bilingual/export")
def bilingual_export(req: BilingualRequest):
    mains, trans, plan = _bilingual_prepare(req)
    blocks = []
    for num, (i, used) in enumerate(plan, 1):
        e = mains[i]
        lines = list(e["lines"])
        for j in used:
            lines.extend(trans[j]["lines"])
        blocks.append(
            f"{num}\n{fmt_time(e['start'])} --> {fmt_time(e['end'])}\n" + "\n".join(lines)
        )
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="bilingual.srt"'},
    )
