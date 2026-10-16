# -*- coding: utf-8 -*-
"""【教師專用】生成：公開測試 student/test_inputs.jsonl、參考答案 dataset.jsonl、隱藏測試 hidden_tests.jsonl。
標籤由 BFS 求解器計算（字典序最小的最短路線，保證唯一）。"""
import json, random, sys
from pathlib import Path
from sokoban import solve, simulate

HERE = Path(__file__).parent
STUDENT = HERE.parent / "student"
sys.path.insert(0, str(STUDENT))
from train import TEST_SYSTEM  # 參考數據的 system 直接沿用測試指令


def random_map(rng, max_side=12, min_side=3):
    h, w = rng.randint(min_side, max_side), rng.randint(min_side, max_side)
    g = [["-"] * w for _ in range(h)]
    if rng.random() < 0.6:  # 帶外牆
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    g[r][c] = "#"
    dens = rng.choice([0, 0.1, 0.2, 0.3])
    for r in range(h):
        for c in range(w):
            if g[r][c] == "-" and rng.random() < dens:
                g[r][c] = "#"
    free = [(r, c) for r in range(h) for c in range(w) if g[r][c] == "-"]
    if len(free) < 3:
        return None
    b = rng.choice(free)
    near = [f for f in free if f != b and abs(f[0] - b[0]) + abs(f[1] - b[1]) <= 3]
    if len(near) < 2:
        return None
    t = rng.choice(near)
    # 注意：玩家站在目標上('+')時，推箱入目標至少需 4 步，故三步題中不會出現 '+'
    p = rng.choice([f for f in near if f != t])
    g[b[0]][b[1]], g[t[0]][t[1]], g[p[0]][p[1]] = "$", ".", "@"
    return ["".join(row) for row in g]


def make(n_by_len, seed, exclude, max_side=12, min_side=3):
    rng = random.Random(seed)
    need, out, seen = dict(n_by_len), [], set(exclude)
    while any(v > 0 for v in need.values()):
        rows = random_map(rng, max_side, min_side)
        if not rows:
            continue
        key = "地圖:\n" + "\n".join(rows)
        route = solve(rows, max_len=3)
        if key in seen or not route or need.get(len(route), 0) <= 0:
            continue
        assert simulate(rows, route)
        need[len(route)] -= 1
        seen.add(key)
        out.append((key, route))
    rng.shuffle(out)
    return out


SIDE = 5  # 固定地圖為 SIDE×SIDE


def dump(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ex = "地圖:\n#####\n#.--#\n#$-##\n#@--#\n#####"
    pub = make({1: 1, 2: 2, 3: 2}, 11, [ex], SIDE, SIDE)  # 題面示例圖留給學生作訓練樣例，不作公開測試
    dump(STUDENT / "test_inputs.jsonl", [{"id": f"M{i:02d}", "input": k, "min_len": len(r)} for i, (k, r) in enumerate(pub, 1)])
    used = [k for k, _ in pub]
    train = make({1: 25, 2: 35, 3: 40}, 2026, used, SIDE, SIDE)
    dump(HERE / "dataset.jsonl", [{"messages": [{"role": "system", "content": TEST_SYSTEM}, {"role": "user", "content": k},
                                                 {"role": "assistant", "content": r}]} for k, r in train])
    hidden = make({1: 1, 2: 2, 3: 2}, 7, used + [k for k, _ in train], SIDE, SIDE)
    dump(HERE / "hidden_tests.jsonl", [{"id": f"H{i:02d}", "input": k, "min_len": len(r)} for i, (k, r) in enumerate(hidden, 1)])
    print(f"public={len(pub)} train={len(train)} hidden={len(hidden)}")


if __name__ == "__main__":
    main()
