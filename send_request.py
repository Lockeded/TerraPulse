import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
import dashscope
from http import HTTPStatus
import base64
from zhipuai import ZhipuAI

with open("sk-zhipu.txt", "r") as f:
    key = f.read().strip()
# 初始化 OpenAI 客户端
# client = OpenAI(
#     api_key=key,
#     base_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
# )
client = ZhipuAI(
    api_key=key
)
def query_gpt(data):
    completion = client.chat.completions.create(
        model="glm-4v-plus-0111",
        messages=data["messages"],
        temperature=0.01,
        seed = 42,
    )
    answer_content = completion.choices[0].message.content
    print(answer_content)
    return answer_content

def main():
    # 读取已处理的 custom_id
    processed_ids = set()
    response_file = "response.jsonl"
    if os.path.exists(response_file):
        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    for custom_id in res.keys():
                        processed_ids.add(custom_id)
                except json.JSONDecodeError:
                    continue

    # 读取 query.jsonl 数据
    with open("query.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    # 以追加模式打开 response.jsonl 文件
    with open(response_file, "a", encoding="utf-8") as f:
        for row in tqdm(data):
            custom_id = row["custom_id"]
            # 检查是否已处理
            if custom_id in processed_ids:
                print(f"Skipping already processed custom_id: {custom_id}")
                continue
            try:
                data_body = row["body"]
                res = query_gpt(data_body)
                # 每次处理完立即写入文件
                response_entry = {custom_id: res}
                f.write(json.dumps(response_entry, ensure_ascii=False) + "\n")
                f.flush()  # 确保写入磁盘
                processed_ids.add(custom_id)  # 更新已处理集合
            except Exception as e:
                print(f"Error processing custom_id {custom_id}: {e}")
                # 记录错误到 error.jsonl
                with open("error.jsonl", "a", encoding="utf-8") as err_f:
                    err_f.write(json.dumps({"custom_id": custom_id, "error": str(e)}, ensure_ascii=False) + "\n")
            break

if __name__ == "__main__":
    main()