import os
from huggingface_hub import hf_hub_download

# 设置代理
proxy = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy

# 下载文件
for i in range(5):
    if i < 5:
        continue
    hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/OpenWorld")
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/OpenWorld")
