import re

def process_data(records):
    """
    每条记录包含 sequence（原括号串）和 valid（已经标注的 YES 或 NO）两个字段，两者都必须是字符串。示例：

    {"sequence":"() ()","valid":" yes "}
    实现 process_data(records)，输入是原始记录列表；返回按原始顺序排列的标准记录列表。上面的记录应整理为：

    [{"input":"序列：()()","output":"[Valid]: YES"}]
    函数只清洗格式并复制你提供的标签，不能数括号、判断合法性或计算标签。输入是否合法和标签是否正确，需要在构造数据时由你确认。

    数据清洗的完整规则
    1. records 不是列表时，抛出 ValueError。按原始顺序处理；不是字典、缺少 sequence/valid 或字段类型不符合的记录，跳过。只读取这两个规定字段，其他字段忽略。
    2. 删除 sequence 中的全部空白字符。处理后长度必须为 2 至 12，而且只能包含半角 ( 和 )，否则跳过整条记录。这里只做字符和长度检查，不判断括号是否合法。
    3. valid 去掉首尾空白并转为大写，只接受 YES 或 NO；否则跳过整条记录。input 为 序列： 加规范化后的括号串；output 为 [Valid]: 加已有标签。
    4. 按规范化后的 input 稳定去重。同一 input 和同一 output 只保留首次出现；同一 input 对应不同 output 时，抛出 ValueError，不能悄悄选一个标签。
    5. 返回列表，不打印答案，不扩增数据。不调用模型、网络或外部服务，不读取文件或评测数据，不用匹配算法补写标签。
    6. 函数代码最多 16KiB，只使用纯函数定义、文档字符串及 json、re、collections 标准库；常量写在函数内。
    公开函数测试可以包含 null 等非对象值，应跳过；正式 train.jsonl 每行仍须为 JSON 对象。函数测试可以返回空列表，但实际训练必须在处理后保留 1 至 100 条数据。
    """
    # 1. 检查 records 是否为列表
    if not isinstance(records, list):
        raise ValueError("records 不是列表")

    valid_records = []
    for r in records:
        # 不是字典、缺少规定字段、或字段类型不是字符串则跳过
        if not isinstance(r, dict):
            continue
        if "sequence" not in r or "valid" not in r:
            continue
        seq = r["sequence"]
        val = r["valid"]
        if not isinstance(seq, str) or not isinstance(val, str):
            continue

        # 2. 删除 sequence 中的全部空白字符，检查长度和字符范围
        clean_seq = re.sub(r'\s+', '', seq)
        if not (2 <= len(clean_seq) <= 12):
            continue
        if not re.fullmatch(r'^[()]+$', clean_seq):
            continue

        # 3. valid 去掉首尾空白并转为大写，只接受 YES 或 NO
        clean_val = val.strip().upper()
        if clean_val not in ("YES", "NO"):
            continue

        input_str = "序列：" + clean_seq
        output_str = "[Valid]: " + clean_val

        valid_records.append({
            "input": input_str,
            "output": output_str
        })

    # 4. 按规范化后的 input 稳定去重，检测冲突并抛出 ValueError
    seen = {}
    result = []
    for item in valid_records:
        inp = item["input"]
        out = item["output"]
        if inp in seen:
            if seen[inp] != out:
                raise ValueError(f"同一 input 对应不同 output: {inp}")
            # 同一 input 且同一 output，保留首次出现，跳过后续重复项
        else:
            seen[inp] = out
            result.append(item)

    return result

def build_training_submission(dataset: str, params: dict) -> dict:
    """
    构建训练提交数据
    :param dataset: 训练数据集，JSONL 格式的字符串
    :param params: 训练参数字典
    :return: 提交数据字典
    """
    params.update({
        "epochs": 1,
        "learningRate": 0.0001,
        "loraRank": 16,
        "batchSize": 1,
        "gradAccum": 8
    })
    return {"dataset": dataset, "params": params}
