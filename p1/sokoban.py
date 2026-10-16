# -*- coding: utf-8 -*-
"""【教師專用】推箱子核心：解析、模擬、最短路線求解（BFS）。"""
from collections import deque

# 展開順序按字母序 D,L,R,U：BFS 首次到達即得「字典序最小的最短路線」，用作唯一標準標籤
DIRS = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}


def parse(rows):
    walls, player, box, target = set(), None, None, None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            if ch in "@+":
                player = (r, c)
            if ch in "$*":
                box = (r, c)
            if ch in ".+*":
                target = (r, c)
    return len(rows), len(rows[0]), walls, player, box, target


def _free(h, w, walls, p):
    return 0 <= p[0] < h and 0 <= p[1] < w and p not in walls


def solve(rows, max_len=None):
    """返回字典序最小的最短路線字符串；無解（或超過 max_len）返回 None；箱子已在目標上返回 ''。"""
    h, w, walls, p, b, t = parse(rows)
    if b == t:
        return ""
    q = deque([(p, b, "")])
    seen = {(p, b)}
    while q:
        p, b, path = q.popleft()
        if max_len is not None and len(path) >= max_len:
            continue
        for m, (dr, dc) in DIRS.items():
            np_ = (p[0] + dr, p[1] + dc)
            if not _free(h, w, walls, np_):
                continue
            nb = b
            if np_ == b:
                nb = (b[0] + dr, b[1] + dc)
                if not _free(h, w, walls, nb):
                    continue
            if (np_, nb) in seen:
                continue
            if nb == t:
                return path + m
            seen.add((np_, nb))
            q.append((np_, nb, path + m))
    return None


def simulate(rows, route):
    """route 合法且最終箱子在目標上 -> True。"""
    h, w, walls, p, b, t = parse(rows)
    for m in route:
        dr, dc = DIRS[m]
        np_ = (p[0] + dr, p[1] + dc)
        if not _free(h, w, walls, np_):
            return False
        if np_ == b:
            nb = (b[0] + dr, b[1] + dc)
            if not _free(h, w, walls, nb):
                return False
            b = nb
        p = np_
    return b == t
