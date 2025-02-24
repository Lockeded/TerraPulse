import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

with open("sk.txt", "r") as f:
    key = f.read().strip()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key = key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def query_gpt(body):
     completion = client.chat.completions.create(body)
     return completion.choices[0].message.content

def main():
    with open("output/queries.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    response = []
    for row in tqdm(data):
        body = row["body"]
        res = query_gpt(body)
        response.append(res)
        time.sleep(1)
    with open("output/response.jsonl", "w", encoding="utf-8") as f:
        for res in response:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        


if __name__ == "__main__":
    main()