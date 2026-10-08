import json
import ast
import re
from pathlib import Path


def parse_maybe_list(x):
    """
    兼容两种情况：
    1. x 本身就是 list
    2. x 是字符串形式的 list，例如 "[1, 2, 3]"
    """
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        s = x.strip()

        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    # 例如 "[signal列表，长度: 1935，显示前20个值]"
                    return x

        return x

    return x


def get_signal_len(signal):
    """
    获取 signal 长度。

    支持：
    1. signal 是 list，例如 [0.1, 0.2, ...]
    2. signal 是占位字符串，例如 "[signal列表，长度: 1935，显示前20个值]"
    """
    if isinstance(signal, list):
        return len(signal)

    if isinstance(signal, str):
        # 尝试从 “长度: 1935” 中解析长度
        match = re.search(r"长度[:：]\s*(\d+)", signal)
        if match:
            return int(match.group(1))

        # 如果是普通字符串形式的 list，但前面没转成功，这里再保底尝试一次
        s = signal.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                pass

    return None


def iter_jsonl_records(jsonl_path):
    """
    逐行读取 jsonl 文件。
    默认假设每一行是一个完整 JSON。
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON 解析失败: {jsonl_path}, line={line_idx}\n"
                    f"请确认 signal_none.jsonl 是否是一行一个 JSON 对象。"
                ) from e


def collect_signal_none_to_jsonl(
    root_dir,
    output_jsonl,
    folder_prefix="LB06",
    save_stats_line=True
):
    """
    合并指定前缀文件夹中的 signal_none.jsonl，并统计 signal 最长和最短长度。

    参数：
    root_dir: result_0331 所在路径
    output_jsonl: 输出 jsonl 文件路径
    folder_prefix: 想合并的文件夹前缀，例如 "LB06" 或 "LB07"
    save_stats_line: 是否在 jsonl 最后一行写入统计信息
    """
    root_dir = Path(root_dir)
    output_jsonl = Path(output_jsonl)

    if not root_dir.exists():
        raise FileNotFoundError(f"找不到目录: {root_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    selected_dirs = sorted([
        p for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith(folder_prefix)
    ])

    print(f"找到 {len(selected_dirs)} 个 {folder_prefix} 开头的文件夹")

    total_records = 0

    min_signal_len = None
    max_signal_len = None

    min_signal_record = None
    max_signal_record = None

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for folder in selected_dirs:
            jsonl_path = folder / "signal_none.jsonl"

            if not jsonl_path.exists():
                print(f"跳过，没有找到文件: {jsonl_path}")
                continue

            print(f"读取: {jsonl_path}")

            file_records = 0

            for obj in iter_jsonl_records(jsonl_path):
                pattern = obj.get("pattern")
                base_sample_spans_rel = parse_maybe_list(
                    obj.get("base_sample_spans_rel")
                )
                signal = parse_maybe_list(
                    obj.get("signal")
                )

                signal_len = get_signal_len(signal)

                new_obj = {
                    "record_type": "sample",
                    "source_folder": folder.name,
                    "read_id": obj.get("read_id"),
                    "signal_key": obj.get("signal_key"),
                    "pattern": pattern,
                    "base_sample_spans_rel": base_sample_spans_rel,
                    "signal": signal,
                    "signal_len": signal_len,
                }

                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

                if signal_len is not None:
                    if min_signal_len is None or signal_len < min_signal_len:
                        min_signal_len = signal_len
                        min_signal_record = {
                            "source_folder": folder.name,
                            "read_id": obj.get("read_id"),
                            "signal_key": obj.get("signal_key"),
                            "signal_len": signal_len,
                        }

                    if max_signal_len is None or signal_len > max_signal_len:
                        max_signal_len = signal_len
                        max_signal_record = {
                            "source_folder": folder.name,
                            "read_id": obj.get("read_id"),
                            "signal_key": obj.get("signal_key"),
                            "signal_len": signal_len,
                        }

                file_records += 1
                total_records += 1

            print(f"  读取并写入记录数: {file_records}")

        stats_obj = {
            "record_type": "stats",
            "folder_prefix": folder_prefix,
            "total_records": total_records,
            "min_signal_len": min_signal_len,
            "max_signal_len": max_signal_len,
            "min_signal_record": min_signal_record,
            "max_signal_record": max_signal_record,
        }

        if save_stats_line:
            fout.write(json.dumps(stats_obj, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"合并前缀: {folder_prefix}")
    print(f"总记录数: {total_records}")
    print(f"最短信号长度: {min_signal_len}")
    print(f"最长信号长度: {max_signal_len}")
    print(f"最短信号记录: {min_signal_record}")
    print(f"最长信号记录: {max_signal_record}")
    print(f"已保存到: {output_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    root_dir = "/mnt/si002562jbsc/rnamodel/wangxue/DNA_modification/03.token/result_0331"

    collect_signal_none_to_jsonl(
        root_dir=root_dir,
        output_jsonl="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB06/lb06_signal_none_selected.jsonl",
        folder_prefix="LB06",
        save_stats_line=True
    )

    # 如果想合并 LB07，改成这样：
    collect_signal_none_to_jsonl(
        root_dir=root_dir,
        output_jsonl="/mnt/si002562jbsc/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/LB07/lb07_signal_none_selected.jsonl",
        folder_prefix="LB07",
        save_stats_line=True
    )