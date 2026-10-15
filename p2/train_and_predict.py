import json, random, re
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from submission import process_data

MODEL, REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
SYSTEM = ("判断序列是否合法：从左到右任意前缀左括号数不能少于右括号，最终两者数目相等。"
          "合法只输出[Valid]: YES，否则只输出[Valid]: NO，冒号后一个半角空格，不加解释。")
PARAMS = {"epochs": range(1, 13), "loraRank": (4, 8, 16), "batchSize": (1,), "gradAccum": (4, 8, 16)}
PUBLIC_FUNC_TESTS = [  # 两个公开数据处理测试点（P01 / P02）
    ([{"sequence": "( )(())", "valid": " yes "}, {"sequence": ")(", "valid": "no"}],
     [{"input": "序列：()(())", "output": "[Valid]: YES"}, {"input": "序列：)(", "output": "[Valid]: NO"}]),
    ([None, {"sequence": " ", "valid": "NO"}, {"sequence": " ( ( ) ) ", "valid": "yes"},
      {"sequence": "(())", "valid": "YES"}, {"sequence": "(a)", "valid": "NO"}, {"sequence": "()", "valid": "MAYBE"}],
     [{"input": "序列：(())", "output": "[Valid]: YES"}]),
]


def check_function():
    for i, (records, want) in enumerate(PUBLIC_FUNC_TESTS, 1):
        try:
            got = process_data(records)
        except Exception as exc:  # noqa: BLE001
            got = f"抛出异常 {exc!r}"
        print(f"P0{i} 数据处理：{'通过' if got == want else '未通过'}" + ("" if got == want else f"\n  期望 {want}\n  实际 {got}"))


def load_training_data(tests):
    raw = open("train.jsonl", "rb").read()
    assert len(raw) <= 256 * 1024, "train.jsonl 超过 256KiB"
    records = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    assert 1 <= len(records) <= 100 and all(isinstance(r, dict) for r in records), "原始数据须为 1–100 个 JSON 对象"
    data = process_data(records)
    assert isinstance(data, list) and 1 <= len(data) <= 100, "处理后数据须为 1–100 条"
    seen = set()
    for d in data:
        assert set(d) == {"input", "output"} and isinstance(d["input"], str), f"记录格式错误：{d!r}"
        assert re.fullmatch(r"序列：[()]{2,12}", d["input"]), f"input 格式错误：{d['input']!r}"
        assert d["output"] in ("[Valid]: YES", "[Valid]: NO"), f"output 格式错误：{d['output']!r}"
        assert d["input"] not in seen, f"重复 input：{d['input']}"
        assert d["input"] not in tests, f"不得把评测输入加入训练集：{d['input']}"
        seen.add(d["input"])
    return data


def load_params():
    p = json.load(open("training_params.json", encoding="utf-8"))
    assert isinstance(p, dict) and set(p) == {"epochs", "learningRate", "loraRank", "batchSize", "gradAccum"}, "参数必须恰好五项"
    for k, allowed in PARAMS.items():
        assert type(p[k]) is int and p[k] in allowed, f"{k} 超出允许范围"
    lr = p["learningRate"]
    assert isinstance(lr, (int, float)) and not isinstance(lr, bool) and 1e-5 <= lr <= 5e-4, "learningRate 超出允许范围"
    return p


def prompt_ids(tok, user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    check_function()
    tests = [json.loads(l) for l in open("test_inputs.jsonl", encoding="utf-8") if l.strip()]
    data = load_training_data({t["input"] for t in tests})
    p = load_params()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=p["loraRank"], lora_alpha=2 * p["loraRank"], lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    opt = torch.optim.AdamW([w for w in model.parameters() if w.requires_grad], lr=p["learningRate"])

    samples = []  # 只对正确答案计算 loss
    for d in data:
        x = prompt_ids(tok, d["input"])
        y = tok(d["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        assert len(x) + len(y) <= 512, f"单条数据超过 512 Token：{d['input']}"
        samples.append((x + y, [-100] * len(x) + y))
    model.train()
    for epoch in range(p["epochs"]):
        random.Random(epoch).shuffle(samples)
        for step, (ids, labels) in enumerate(samples, 1):  # batchSize 固定为 1
            out = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([labels], device=dev))
            (out.loss / p["gradAccum"]).backward()
            if step % p["gradAccum"] == 0 or step == len(samples):
                opt.step()
                opt.zero_grad()
        print(f"epoch {epoch + 1}/{p['epochs']} 完成")

    model.save_pretrained("lora_final")  # 保存并重新加载最后一轮 LoRA
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(dev)
    model = PeftModel.from_pretrained(base, "lora_final").to(dev).eval()
    with open("predictions.jsonl", "w", encoding="utf-8") as fout:
        for t in tests:
            x = prompt_ids(tok, t["input"])
            with torch.no_grad():
                out = model.generate(torch.tensor([x], device=dev), max_new_tokens=128, do_sample=False)
            value = tok.decode(out[0][len(x):], skip_special_tokens=True)
            fout.write(json.dumps({"id": t["id"], "value": value}, ensure_ascii=False) + "\n")
            print(t["id"], t["input"], "→", value.strip())
    print("已写出 predictions.jsonl")


if __name__ == "__main__":
    main()
