# -*- coding: utf-8 -*-
"""【教師專用】用法：python grade.py <學生目錄>
學生目錄需含 submission.py、dataset.jsonl；若有 predictions.jsonl（訓練後預測隱藏點）則一併判分。
--selftest：檢驗校驗器能攔住各類壞數據。"""
import json, sys, importlib.util
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "student"))
import train as T  # 與學生同一份校驗/判題邏輯


def selftest():
    good = lambda m, r: json.dumps({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": m}, {"role": "assistant", "content": r}]}, ensure_ascii=False)
    A, B = "地圖:\n#####\n#.--#\n#$-##\n#@--#\n#####", "地圖:\n#####\n#@$.#\n#####"
    cases = {
        "合法": ([good(A, "U"), good(B, "R")], True),
        "合法+think": ([good(A, "<think>推上去</think>U"), good(B, "R")], True),
        "只有1條": ([good(A, "U")], False),
        "路線非法": ([good(A, "D"), good(B, "R")], False),
        "合法但非最短": ([good(A, "U"), good(B, "RL")], False),
        "小寫": ([good(A, "u"), good(B, "R")], False),
        "零步地圖": ([good(A, "U"), good("地圖:\n###\n#*@\n###", "L")], False),
        "冒號後多空格": ([good(A.replace(":\n", ": \n"), "U"), good(B, "R")], False),
        "重複地圖": ([good(A, "U"), good(A, "U")], False),
        "兩個玩家": ([good(A, "U"), good("地圖:\n#####\n#@$.@\n#####", "R")], False),
        "101條": ([good(A, "U")] * 101, False),
    }
    for name, (lines, want) in cases.items():
        try: T.validate_dataset("\n".join(lines)); got = True
        except ValueError: got = False
        print(f"數據 {name}: {'通過' if got == want else '【異常】'}")
    ok = dict(T.DEFAULT_PARAMS)
    for name, mut, want in [("默認", {}, True), ("lr過大", {"learningRate": 0.01}, False), ("rank=12", {"loraRank": 12}, False),
                            ("字符串", {"epochs": "8"}, False), ("多鍵", {"x": 1}, False), ("batch=3", {"batchSize": 3}, False)]:
        try: T.validate_params({**ok, **mut}); got = True
        except ValueError: got = False
        print(f"參數 {name}: {'通過' if got == want else '【異常】'}")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    d = Path(sys.argv[1] if len(sys.argv) > 1 else HERE)
    sys.path.insert(0, str(d)); sys.modules.pop("submission", None)
    pub = {json.loads(l)["input"] for l in open(HERE.parent / "student/test_inputs.jsonl", encoding="utf-8") if l.strip()}
    try:
        recs, p = T.run_submission(open(d / "dataset.jsonl", encoding="utf-8").read(), pub)
        print(f"函數與數據校驗通過：{len(recs)} 條，參數 {p}")
    except Exception as e:  # noqa: BLE001
        print("校驗未通過：", e); return
    pred = d / "predictions.jsonl"
    if pred.exists():
        P = {j["id"]: j["value"] for j in map(json.loads, open(pred, encoding="utf-8"))}
        hs = [json.loads(l) for l in open(HERE / "hidden_tests.jsonl", encoding="utf-8") if l.strip()]
        ok = 0
        for t in hs:
            g = T.judge(t["input"].split("\n")[1:], P.get(t["id"], ""), t["min_len"]); ok += g
            print(t["id"], "通過" if g else "未通過")
        print(f"隱藏模型點 {ok}/{len(hs)}")


if __name__ == "__main__":
    main()
