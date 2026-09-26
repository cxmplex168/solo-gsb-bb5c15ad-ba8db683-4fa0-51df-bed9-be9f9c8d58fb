"""风险复审片段：圈定修订稿中的风险字符并选取覆盖风险最多的互不重叠片段。

处理流程：
1. 复用逐字稿输入校验与逐字时间轴；
2. 风险字符：source=computed 的新增字符，或 source=matched 但修订字符与
   识别字符不同（人工替换纠错）的字符；
3. 把连续的风险字符合并为风险块，每块取首字 start、末字 end，向两侧各
   扩展 padding_ms 并裁剪到录音边界 [0, total_duration]，得到一个候选片段；
4. 从全部候选中选择最多 max_clips 个**互不重叠**（半开区间，端点相接合法）
   的片段，依次最大化：覆盖风险字符总数、最小化播放总时长、按起点排序的
   区间序列字典序；
5. 按时间顺序返回所选片段的录音毫秒区间、风险块半开字符下标与风险字符数。
"""
from __future__ import annotations

from bisect import bisect_right

from .schemas import (
    ReviewClip,
    ReviewClipsRequest,
    RevisedCharTimeline,
)

_EMPTY = -1  # 链空标记


def _is_risky(ch: RevisedCharTimeline) -> bool:
    """新增（computed）或被人工替换（matched 但字符不同）即为风险字符。"""
    return ch.source == "computed" or (
        ch.source == "matched" and ch.recognized_char != ch.char)


def _risk_blocks(chars: list[RevisedCharTimeline]) -> list[tuple[int, int]]:
    """把连续风险字符合并为半开字符区间（按原文顺序）。"""
    blocks: list[tuple[int, int]] = []
    run_start: int | None = None
    for j, ch in enumerate(chars):
        if _is_risky(ch):
            if run_start is None:
                run_start = j
        elif run_start is not None:
            blocks.append((run_start, j))
            run_start = None
    if run_start is not None:
        blocks.append((run_start, len(chars)))
    return blocks


def _candidate_intervals(chars: list[RevisedCharTimeline],
                         blocks: list[tuple[int, int]], padding_ms: int,
                         total_duration: int) -> list[tuple[int, int]]:
    """每个风险块取首字 start、末字 end，扩 padding 后裁剪到录音边界。"""
    intervals: list[tuple[int, int]] = []
    for start_index, end_index in blocks:
        raw_start = chars[start_index].start
        raw_end = chars[end_index - 1].end
        intervals.append((
            max(0, raw_start - padding_ms),
            min(total_duration, raw_end + padding_ms),
        ))
    return intervals


def _select_clips(weights: list[int],
                  intervals: list[tuple[int, int]],
                  max_clips: int) -> list[int]:
    """带基数限制的加权区间调度，返回所选候选下标（升序）。

    正向 DP：``cov/dur/head[k][i]`` 表示只在前 ``i`` 个候选（0..i-1）中至多
    选 ``k`` 个时的最优方案。目标依次为：覆盖风险字符总数最大、播放总时长
    最短、按 ``(起点, 终点)`` 的区间序列字典序最小。方案链用不可变全局节点
    池表达：“取候选 c”即在最优前缀链尾追加节点 c，方案间共享前缀链；仅当
    覆盖数与时长都相同、确需字典序决胜时才分配节点，避免无谓分配。
    """
    m = len(intervals)
    if m == 0:
        return []

    ends = [end for _, end in intervals]
    # compat_count[c]：与候选 c 兼容（end <= start_c）的更早候选个数，即这些
    # 候选恰为前 compat_count[c] 个，对应 DP 前缀列下标。半开区间端点相接
    # 不算重叠；候选起点、终点均非递减，可用二分计数。
    compat_count = [
        bisect_right(ends, intervals[c][0], hi=c) for c in range(m)
    ]

    width = m + 1
    cov = [[0] * width]
    dur = [[0] * width]
    head = [[_EMPTY] * width]
    # 不可变链节点池：node_cand 为节点追加的候选下标，node_next 指向链上前
    # 一节点（下标更小的链尾）；链头必为最新、最大的候选下标。
    pool_cand: list[int] = []
    pool_next: list[int] = []

    def chain_candidates(node: int) -> list[int]:
        out: list[int] = []
        while node != _EMPTY:
            out.append(pool_cand[node])
            node = pool_next[node]
        out.reverse()
        return out

    def lexicographically_less(t_head: int, b_head: int) -> bool:
        t_seq = [intervals[c] for c in chain_candidates(t_head)]
        b_seq = [intervals[c] for c in chain_candidates(b_head)]
        return t_seq < b_seq

    for _ in range(1, max_clips + 1):
        prev_cov, prev_dur, prev_head = cov[-1], dur[-1], head[-1]
        cur_cov = [0] * width
        cur_dur = [0] * width
        cur_head = [_EMPTY] * width
        for i in range(1, width):
            c = i - 1
            # 不取候选 c：直接沿用前 c 个候选上的最优方案。
            skip_cov, skip_dur, skip_head = (
                cur_cov[c], cur_dur[c], cur_head[c])
            # 取候选 c：与其兼容的更早候选只能来自前 compat_count[c] 个。
            # 追加节点（取方案专用、未被任何已有状态共享）；落败时直接回收。
            q = compat_count[c]
            take_node = len(pool_cand)
            pool_cand.append(c)
            pool_next.append(prev_head[q])
            take_cov = prev_cov[q] + weights[c]
            take_dur = prev_dur[q] + intervals[c][1] - intervals[c][0]

            if take_cov > skip_cov or (
                    take_cov == skip_cov and take_dur < skip_dur):
                take_wins = True
            elif take_cov < skip_cov or take_dur > skip_dur:
                take_wins = False
            else:
                take_wins = lexicographically_less(take_node, skip_head)

            if take_wins:
                cur_cov[i], cur_dur[i], cur_head[i] = (
                    take_cov, take_dur, take_node)
            else:
                pool_cand.pop()
                pool_next.pop()
                cur_cov[i], cur_dur[i], cur_head[i] = (
                    skip_cov, skip_dur, skip_head)
        cov.append(cur_cov)
        dur.append(cur_dur)
        head.append(cur_head)

    return chain_candidates(head[max_clips][m])


def build_review_clips(req: ReviewClipsRequest,
                       chars: list[RevisedCharTimeline]) -> list[ReviewClip]:
    """圈定风险块并选出覆盖风险字符最多的互不重叠候选片段。"""
    blocks = _risk_blocks(chars)
    if not blocks:
        return []

    intervals = _candidate_intervals(
        chars, blocks, req.padding_ms, req.total_duration)
    weights = [end - start for start, end in blocks]

    chosen = _select_clips(weights, intervals, req.max_clips)

    return [
        ReviewClip(
            start=intervals[i][0],
            end=intervals[i][1],
            start_index=blocks[i][0],
            end_index=blocks[i][1],
            risk_count=weights[i],
        )
        for i in chosen
    ]
