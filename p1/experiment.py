# -*- coding: utf-8 -*-
"""【教師專用】診斷實驗：到底是數據、超參，還是模型/表示的問題？
用法示例（均在 teacher/ 下運行，需要 train.py、test_inputs.jsonl 在 ../student/）：
  python experiment.py --dry-run                       # 只生成並校驗數據，不訓練
  python experiment.py --train-n 100 --epochs 10 --eval-every 2          # 基線 + 留出集曲線
  python experiment.py --train-n 400  --epochs 5 --eval-every 1          # 數據規模曲線
  python experiment.py --train-n 1600 --epochs 3 --eval-every 1
  python experiment.py --train-n 100 --max-side 7 ...                    # 小地圖分佈
輸出：留出集（默認 60 張全新地圖）準確率，按最短步數分組；另報「走法合法且到達目標（不要求最短）」與「訓練集前 30 張」準確率。"""
import argparse, json, random, re, sys, time
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "student"))
import train as T
from gen_dataset import make


def quotas(n, w=(25, 35, 40)):
    a, b = round(n * w[0] / sum(w)), round(n * w[1] / sum(w))
    return {1: a, 2: b, 3: n - a - b}


def build_data(args):
    pub = [json.loads(l) for l in open(HERE.parent / "student/test_inputs.jsonl", encoding="utf-8") if l.strip()]
    pub_keys = [t["input"] for t in pub]
    tr = make(quotas(args.train_n, tuple(int(x) for x in args.quota.split(','))), args.seed, pub_keys, args.max_side, args.min_side)
    ev = make(quotas(args.eval_n), 999, pub_keys + [k for k, _ in tr], args.eval_max_side, args.min_side)  # 留出集固定 25/35/40 配比
    pub_items = [(t["input"], "U" * t["min_len"]) for t in pub]       # 只用其長度做判分
    return tr, ev, pub_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=100); ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=1); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=12); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval-max-side", type=int, default=12, help="留出集地圖的最大邊長")
    ap.add_argument("--quota", default="25,35,40", help="訓練集 1/2/3 步的比例，如 10,35,55")
    ap.add_argument("--min-side", type=int, default=3, help="訓練/留出地圖的最小邊長")
    ap.add_argument("--fixed-side", type=int, default=0, help="地圖固定為 N×N（訓練與留出都用；公開5點分佈不同，可忽略）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.fixed_side:
        args.min_side = args.max_side = args.eval_max_side = args.fixed_side
    tr, ev, pub_items = build_data(args)
    # 用與學生相同的校驗器檢查生成的數據（本地實驗不受 100 條上限限制）
    T.MAX_RECORDS = max(T.MAX_RECORDS, args.train_n)
    lines = [json.dumps({"messages": [{"role": "system", "content": T.TEST_SYSTEM}, {"role": "user", "content": k}, {"role": "assistant", "content": r}]}, ensure_ascii=False) for k, r in tr]
    T.validate_dataset("\n".join(lines))
    print(f"訓練 {len(tr)} 張、留出 {len(ev)} 張；訓練圖平均 {sum(len(k.split(chr(10)))-1 for k,_ in tr)/len(tr):.1f} 行")
    if args.dry_run:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(T.SEED); torch.manual_seed(T.SEED)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(T.MODEL, revision=T.REVISION)
    model = AutoModelForCausalLM.from_pretrained(T.MODEL, revision=T.REVISION, dtype=dtype).to(dev)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [w for w in model.parameters() if w.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def render(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    samples = []
    for k, r in tr:
        x = tok(render(T.TEST_SYSTEM, k), add_special_tokens=False)["input_ids"]
        y = tok(r + "<|im_end|>", add_special_tokens=False)["input_ids"]
        samples.append((x + y, [-100] * len(x) + y))

    def evaluate(items, bs=16):
        model.eval(); tok.padding_side = "left"; res = []
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            enc = tok([render(T.TEST_SYSTEM, k) for k, _ in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
            for (k, r), o in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                v = tok.decode(o, skip_special_tokens=True); rows = k.split("\n")[1:]
                fmt = bool(re.fullmatch(r"[UDLR]{1,30}", v))
                res.append((len(r), T.judge(rows, v, len(r)), fmt and T.simulate(rows, v), v))
        model.train()
        return res

    def report(tag, res):
        n = len(res); by = {L: [x for x in res if x[0] == L] for L in (1, 2, 3)}
        per = " ".join(f"{L}步 {sum(x[1] for x in v)}/{len(v)}" for L, v in by.items() if v)
        print(f"  [{tag}] 精確(最短) {sum(x[1] for x in res)}/{n} | 到達目標(不要求最短) {sum(x[2] for x in res)}/{n} | {per}")

    def collate(batch):
        n = max(len(i) for i, _ in batch); t = lambda v: torch.tensor(v, device=dev)
        return (t([i + [tok.pad_token_id] * (n - len(i)) for i, _ in batch]), t([l + [-100] * (n - len(l)) for _, l in batch]),
                t([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]))

    model.train(); t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        random.Random(epoch - 1).shuffle(samples)
        batches = [samples[i:i + args.bs] for i in range(0, len(samples), args.bs)]
        total = 0.0
        for step, batch in enumerate(batches, 1):
            ids, lab, att = collate(batch)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / args.accum).backward(); total += loss.item()
            if step % args.accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        print(f"epoch {epoch}/{args.epochs} loss={total / len(batches):.4f} ({time.time() - t0:.0f}s)")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            report("留出集", evaluate(ev)); report("公開5點", evaluate(pub_items)); report("訓練集前30", evaluate(tr[:30]))


if __name__ == "__main__":
    main()
