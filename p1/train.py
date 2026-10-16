# -*- coding: utf-8 -*-
"""【學生可見】完整訓練腳本：讀取 dataset.jsonl -> 運行 build_training_submission -> 校驗 -> LoRA 訓練 -> 重載預測 -> 判分。
固定設置（題面）：seed 2026，僅 assistant token 計 loss，單條 ≤4096 token，LoRA 目標 7 個投影層，
alpha=2*rank，dropout=0.05，AdamW 權重衰減 0.01，梯度裁剪 1.0。"""
import json, random, re
from collections import deque

MODEL, REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
# 測試階段的固定指令（只讀；訓練數據中的 system.content 不會替換它）
TEST_SYSTEM = ("你是單箱推箱子求解器。地圖符號：#牆，-空地，@玩家，$箱子，.目標，+玩家在目標上，*箱子在目標上。"
               "U/D/L/R 表示玩家向上/下/左/右走一格；走向箱子時把箱子推前一格，牆和邊界不能進入，箱子不能被推入牆或出界。"
               "只輸出把箱子推到目標的最短路線，由連續大寫 U D L R 組成，不要空格、換行、標點或解釋。")
SEED, MAX_TOKENS, MIN_RECORDS, MAX_RECORDS = 2026, 4096, 2, 100
DEFAULT_PARAMS = {"epochs": 8, "learningRate": 0.0003, "loraRank": 16, "batchSize": 1, "gradAccum": 8}
PARAM_CHOICES = {"epochs": range(1, 21), "loraRank": (4, 8, 16, 32), "batchSize": (1, 2, 4), "gradAccum": (1, 2, 4, 8, 16)}
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


# ---------------- 地圖解析、求解與判題（規則公開） ----------------
def parse_user_content(text):
    """user.content 必須是 '地圖:\\n' + 逐行網格；返回行列表，不合規拋 ValueError。"""
    if not isinstance(text, str) or not text.startswith("地圖:\n"):
        raise ValueError("user.content 須以 '地圖:' 加一個換行開頭")
    rows = text[len("地圖:\n"):].split("\n")
    if not 3 <= len(rows) <= 12 or not 3 <= len(rows[0]) <= 12 or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("地圖須為 3-12 行、3-12 列且各行等寬")
    s = "".join(rows)
    if not set(s) <= set("#-@$.+*"):
        raise ValueError("地圖含非法字符")
    if s.count("@") + s.count("+") != 1 or s.count("$") + s.count("*") != 1 or s.count(".") + s.count("+") + s.count("*") != 1:
        raise ValueError("地圖須恰好 1 個玩家、1 個箱子、1 個目標")
    return rows


def _locate(rows):
    cells = {(r, c): ch for r, row in enumerate(rows) for c, ch in enumerate(row)}
    pick = lambda chars: next(k for k, v in cells.items() if v in chars)
    return cells, pick("@+"), pick("$*"), pick(".+*")


def _step(cells, p, b, m):
    """走一步 m；非法返回 None，否則返回 (新玩家, 新箱子)。"""
    d = DIRS[m]
    q = (p[0] + d[0], p[1] + d[1])
    if cells.get(q, "#") == "#":
        return None
    if q == b:
        nb = (b[0] + d[0], b[1] + d[1])
        if cells.get(nb, "#") == "#":
            return None
        return q, nb
    return q, b


def simulate(rows, route):
    """路線合法且最終箱子在目標上。"""
    cells, p, b, t = _locate(rows)
    for m in route:
        nxt = _step(cells, p, b, m)
        if nxt is None:
            return False
        p, b = nxt
    return b == t


def shortest_len(rows):
    """BFS 求最短步數；箱子已在目標上返回 0，無解返回 None。"""
    cells, p, b, t = _locate(rows)
    if b == t:
        return 0
    q, seen = deque([(p, b, 0)]), {(p, b)}
    while q:
        p, b, n = q.popleft()
        for m in "UDLR":
            nxt = _step(cells, p, b, m)
            if nxt is None or nxt in seen:
                continue
            if nxt[1] == t:
                return n + 1
            seen.add(nxt)
            q.append((*nxt, n + 1))
    return None


def judge(rows, value, min_len):
    """測試輸出判分：原樣匹配 [UDLR]{1,30}（不去空白），合法、到達目標且長度等於最短。"""
    return bool(re.fullmatch(r"[UDLR]{1,30}", value)) and simulate(rows, value) and len(value) == min_len


# ---------------- 數據與參數校驗 ----------------
def validate_dataset(text, forbidden=()):
    """校驗 dataset 文本，返回 [(system, rows, assistant)]；任何問題拋 ValueError（帶行號）。"""
    lines = [l for l in text.splitlines() if l.strip()]
    if not MIN_RECORDS <= len(lines) <= MAX_RECORDS:
        raise ValueError(f"數據須為 {MIN_RECORDS}-{MAX_RECORDS} 條，當前 {len(lines)} 條")
    out, seen = [], set()
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            msgs = obj["messages"]
            if not (isinstance(msgs, list) and len(msgs) == 3 and [m.get("role") for m in msgs] == ["system", "user", "assistant"]
                    and all(isinstance(m.get("content"), str) for m in msgs)):
                raise ValueError("messages 須為 system/user/assistant 三條且 content 為字符串")
            rows = parse_user_content(msgs[1]["content"])
            m = re.fullmatch(r"(?:<think>.*?</think>\s*)?([UDLR]{1,30})", msgs[2]["content"], re.S)
            if not m:
                raise ValueError("assistant.content 須為 1-30 位大寫 UDLR（可前置 <think>…</think>）")
            route, best = m.group(1), shortest_len(rows)
            if best == 0:
                raise ValueError("箱子已在目標上的零步地圖不接收")
            if best is None:
                raise ValueError("地圖無解")
            if not simulate(rows, route):
                raise ValueError("路線不合法或未把箱子推到目標")
            if len(route) != best:
                raise ValueError(f"路線不是最短（長度 {len(route)}，最短 {best}）")
            if msgs[1]["content"] in seen:
                raise ValueError("重複地圖")
            if msgs[1]["content"] in forbidden:
                raise ValueError("不得把公開測試地圖加入訓練集")
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"第 {i} 行：{e}") from None
        seen.add(msgs[1]["content"])
        out.append((msgs[0]["content"], rows, msgs[2]["content"]))
    return out


def validate_params(p):
    if not isinstance(p, dict) or set(p) != set(DEFAULT_PARAMS):
        raise ValueError("params 必須恰好包含五個鍵")
    for k, allowed in PARAM_CHOICES.items():
        if type(p[k]) is not int or p[k] not in allowed:
            raise ValueError(f"{k} 須為允許的整數值")
    lr = p["learningRate"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 1e-5 <= lr <= 1e-3:
        raise ValueError("learningRate 須在 0.00001-0.001")


def run_submission(dataset_text, forbidden=()):
    """運行學生函數並校驗其返回值；返回 (records, params)。"""
    from submission import build_training_submission
    res = build_training_submission(dataset_text, dict(DEFAULT_PARAMS))
    if not isinstance(res, dict) or set(res) != {"dataset", "params"} or not isinstance(res["dataset"], str):
        raise ValueError('函數須返回 {"dataset": 字符串, "params": 字典}')
    validate_params(res["params"])
    return validate_dataset(res["dataset"], forbidden), res["params"]


# ---------------- 訓練與預測 ----------------
def main():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tests = [json.loads(l) for l in open("test_inputs.jsonl", encoding="utf-8") if l.strip()]
    records, p = run_submission(open("dataset.jsonl", encoding="utf-8").read(), {t["input"] for t in tests})
    print(f"數據校驗通過：{len(records)} 條；參數 {p}")
    random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=p["loraRank"], lora_alpha=2 * p["loraRank"], lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [w for w in model.parameters() if w.requires_grad]
    opt = torch.optim.AdamW(params, lr=p["learningRate"], weight_decay=0.01)

    def prompt_ids(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    samples = []  # 只對 assistant 部分計算 loss
    for k, (system, rows, answer) in enumerate(records, 1):
        x = prompt_ids(system, "地圖:\n" + "\n".join(rows))
        y = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(x) + len(y) > MAX_TOKENS:
            raise ValueError(f"第 {k} 條超過 {MAX_TOKENS} token，不自動截斷")
        samples.append((x + y, [-100] * len(x) + y))

    def collate(batch):  # 右側補齊
        n = max(len(i) for i, _ in batch)
        pad = tok.pad_token_id
        t = lambda v: torch.tensor(v, device=dev)
        return (t([i + [pad] * (n - len(i)) for i, _ in batch]), t([l + [-100] * (n - len(l)) for _, l in batch]),
                t([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]))

    model.train()
    bs, ga, log = p["batchSize"], p["gradAccum"], []
    for epoch in range(p["epochs"]):
        random.Random(epoch).shuffle(samples)
        batches = [samples[i:i + bs] for i in range(0, len(samples), bs)]
        total = 0.0
        for step, batch in enumerate(batches, 1):
            ids, lab, att = collate(batch)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / ga).backward(); total += loss.item()
            if step % ga == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
        log.append({"epoch": epoch + 1, "loss": total / len(batches)})
        print(f"epoch {epoch + 1}/{p['epochs']} loss={log[-1]['loss']:.4f}")

    model.save_pretrained("lora_final"); tok.save_pretrained("lora_final")
    json.dump({"params": p, "log": log}, open("lora_final/training_log.json", "w"), ensure_ascii=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(dev)
    model = PeftModel.from_pretrained(base, "lora_final").to(dev).eval()
    good = 0
    with open("predictions.jsonl", "w", encoding="utf-8") as fout:
        for t in tests:
            x = prompt_ids(TEST_SYSTEM, t["input"])
            with torch.no_grad():
                out = model.generate(torch.tensor([x], device=dev), max_new_tokens=64, do_sample=False)
            value = tok.decode(out[0][len(x):], skip_special_tokens=True)
            ok = judge(t["input"].split("\n")[1:], value, t["min_len"]); good += ok
            fout.write(json.dumps({"id": t["id"], "value": value}, ensure_ascii=False) + "\n")
            print(t["id"], "通過" if ok else "未通過", "→", repr(value))
    print(f"公開模型點 {good}/{len(tests)}；已寫出 predictions.jsonl")


if __name__ == "__main__":
    main()
