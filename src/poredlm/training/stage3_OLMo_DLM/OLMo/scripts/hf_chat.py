import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="单卡聊天脚本")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Hugging Face 格式的模型路径，例如：/mnt/nju/olmorun/hf_converted_olmo"
    )
    args = parser.parse_args()

    # 检查 CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 本脚本需要 CUDA 支持，但未检测到 GPU。")
    device = "cuda"
    print(f"🎮 使用 GPU: {torch.cuda.get_device_name(0)}")

    # 加载 tokenizer 和模型
    print(f"🚀 正在加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # 或 torch.float16
    ).to(device)

    model.eval()  # 设置为评估模式

    print(f"✅ 模型已加载，运行在: {device}")
    print("\n💬 开始聊天！输入 'q' 退出")

    # 聊天主循环
    while True:
        user_input = input("\n🧑💻 你: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 构造输入（你可以在这里添加 system prompt）
        prompt = user_input

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id  # 防止 warning
            )

        # 解码回复（跳过输入部分）
        reply_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True)

        print(f"\n🤖 模型: {reply}")

if __name__ == "__main__":
    main()
