import json
import random
from pathlib import Path

# 正确答案由规则计算：左括号 +1 ，右括号 -1，任何前缀的和不能为负数，最终和必须为零。
def is_valid(sequence: str) -> bool:
    balance = 0
    for char in sequence:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        if balance < 0:
            return False
    return balance == 0

def main():
    # 生成训练数据：随机生成括号序列，计算正确答案。
    rng = random.Random()
    valid, invalid = [], []
    while len(valid) < 40 or len(invalid) < 40:
        length = rng.randint(2, 12)  # 序列长度为 2 到 12
        sequence = "".join(rng.choice("()") for _ in range(length))
        group = valid if is_valid(sequence) else invalid
        if len(group) < 40 and sequence not in group:
            group.append(sequence)

    sequences = valid + invalid
    rng.shuffle(sequences)

    # 保存训练数据到 JSONL 文件，每行包含 sequence 和 valid 字段
    output = Path(__file__).with_name("train.jsonl")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for seq in sequences:
            val_str = "YES" if is_valid(seq) else "NO"
            sample = {
                "sequence": seq,
                "valid": val_str
            }
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"生成训练数据 {len(sequences)} 条，保存到 {output}")

if __name__ == "__main__":
    main()
