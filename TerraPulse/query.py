import os
import torch
import faiss
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from openai import OpenAI
from .classify import classify
from pathlib import Path
import pandas as pd

with open("sk.txt", "r") as f:
        key = f.read().strip()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key = key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 全局设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 防止冲突报错
embedding_path = r"F:/clip_embeddings.pt"
embedding_data = torch.load(embedding_path, weights_only=True)


def load_embeddings() -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """加载嵌入数据。"""
    image_embeddings_list = []
    text_list = []
    image_id_list = []
    cell_id_list = []

    image_embeddings_list = (embedding_data["image"].cpu().numpy())
    text_list = (embedding_data["text"])
    image_id_list = (embedding_data["image_id"])
    cell_id_list = (embedding_data["cell_id"])

    return image_embeddings_list, text_list, image_id_list, cell_id_list

def filter_embeddings(classify_df, target_p_key: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    global embedding_data
    """根据地理层级过滤匹配cell_id的嵌入数据"""
    # 地理层级映射字典
    target_p_key_dict = {"coarse": 0, "middle": 1, "fine": 2, "hierarchy": 2}
    
    # 验证输入参数有效性
    if target_p_key_dict.get(target_p_key) is None:
        raise ValueError(f"无效的target_p_key: {target_p_key}，有效值为{list(target_p_key_dict.keys())}")
    
    # 获取目标层级索引和对应的预测类别
    level_index = target_p_key_dict[target_p_key]
    target_pred_classes = str(classify_df[classify_df['p_key'] == target_p_key]['pred_class'].tolist()[0])
    
    # 初始化存储容器
    filtered_embeddings = []
    filtered_texts = []
    filtered_image_ids = []
    filtered_cell_ids = []
    
    # 遍历所有样本进行过滤
    for i in range(len(embedding_data["image"])):
        # 获取当前样本的cell_id层级值
        current_cell_id = embedding_data["cell_id"][i]
        
        # 跳过无效的cell_id数据
        if current_cell_id is None or len(current_cell_id) <= level_index:
            continue
            
        # 提取对应层级的cell_id值
        level_cell_id = current_cell_id[level_index]
        # 进行匹配过滤
        if str(level_cell_id) == target_pred_classes:
            filtered_embeddings.append(embedding_data["image"][i].numpy())
            filtered_texts.append(embedding_data["text"][i])
            filtered_image_ids.append(embedding_data["image_id"][i])
            filtered_cell_ids.append(embedding_data["cell_id"][i])
    # 转换为numpy数组（如果存在有效数据）
    final_embeddings = np.stack(filtered_embeddings) if filtered_embeddings else np.array([])
    
    return (
        final_embeddings,
        filtered_texts,
        filtered_image_ids,
        [cid[level_index] for cid in filtered_cell_ids]  # 只返回目标层级的cell_id
    )

def build_faiss_index(image_embeddings):      # 预期 (n, d)
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # 应保持 (n, d)
    index = faiss.IndexFlatL2(image_embeddings_norm.shape[1])
    index.add(image_embeddings_norm)
    return index

def get_image_embedding(image_path: str) -> np.ndarray:
    """提取单张查询图片的嵌入特征。"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

def classify_s2cell(image_path: str) -> str:
    """使用分类模型预测图片的 H3 单元 ID。"""
    checkpoint_path = Path("TerraPulse/models/base_M/epoch=014-val_loss=18.4833.ckpt")
    hparams_path = Path("TerraPulse/models/base_M/hparams.yaml")
    use_gpu = False  # Set to False if you want to run on CPU
    image_path = Path(image_path)  # Path to the image you want to predict
    df = classify(checkpoint_path, hparams_path, image_path, use_gpu, is_single_image=True)
    print(df)
    return df
    

def query_gpt_with_images(
    query_image_path: str,
    prompt: str, 
) -> str:
    """使用 大模型 查询图片地理坐标。"""
    query_image_name = query_image_path.stem + query_image_path.suffix
    completion = client.chat.completions.create(
        model="qwen-vl-plus-latest",
        seed=42,
        messages=[
                {"role": "system", "content": "You are an expert in image-based geolocation. Your task is to analyze query images, consider reference coordinates, and estimate their most likely geographic location. Always output coordinates in (latitude, longitude) format, e.g., (25.745593, -80.177536), without any additional explanation."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"http://www.lockede.me:5000/image/{query_image_name}"},
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ],
    timeout=60,
    temperature=0.7,
    )
    
    return completion.choices[0].message.content

import json
from pathlib import Path

def query_gpt_with_images_to_jsonl_append(
    query_image_path: Path,
    prompt: str,
    jsonl_file: Path,
) -> None:
    """使用大模型查询图片地理坐标，读取现有jsonl文件后续写新请求数据。"""
    # 读取现有的 JSONL 文件内容
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            existing_requests = [json.loads(line.strip()) for line in f]
    else:
        existing_requests = []  # 如果文件不存在，初始化为空列表

    query_image_name = query_image_path.stem + query_image_path.suffix
    request_data = {
        "custom_id": query_image_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "qwen-vl-plus",
            "seed": 42,
            "timeout": 60,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "You are an expert in image-based geolocation. Your task is to analyze query images, consider reference coordinates, and estimate their most likely geographic location. Always output coordinates in (latitude, longitude) format, e.g., (25.745593, -80.177536), without any additional explanation."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"http://www.lockede.me:5000/image/{query_image_name}"},
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
        }
    }
    # 将新请求数据追加到现有数据列表
    existing_requests.append(request_data)

    # 重新写入更新后的数据到 JSONL 文件
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for request in existing_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def search_most_similar_images(
    query_embedding: np.ndarray, 
    index: faiss.IndexFlatIP, 
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 FAISS 索引查找最相似的 K 个图片，返回距离和索引。"""
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(query_embedding_norm, k)
    return distances, indices

def search_less_similar_images(
    query_embedding: np.ndarray, 
    index: faiss.IndexFlatIP, 
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 FAISS 索引查找最相似的 K 个图片，返回距离和索引。"""
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(-query_embedding_norm, k)
    return -distances, indices

def get_coordinates_list(
        indices: np.ndarray, 
        text_list: List[str]
) -> List[str]:
    coordinates_list = []
    for i in range(len(indices[0])):
        coordinates = text_list[indices[0][i]]
        coordinates_list.append(coordinates)
    return coordinates_list

def display_results(
    query_image_path: str, 
    indices: np.ndarray, 
    distances: np.ndarray, 
    image_id_list: List[str], 
    text_list: List[str], 
    cell_id_list: List[str]
) -> None:
    """输出最相似的图片的 ID、坐标、相似度和 H3 单元 ID。"""
    print(f"查询图片: {query_image_path}")
    print(f"找到与查询图片最相似的 {len(indices[0])} 张图片：")
    for i in range(len(indices[0])):
        image_id = image_id_list[indices[0][i]].replace("/", "_")
        coordinates = text_list[indices[0][i]]
        cell_id = cell_id_list[indices[0][i]]
        print(f"图片 ID: {image_id}, 坐标: {coordinates}, 相似度: {distances[0][i]:.4f}, H3 单元: {cell_id}")


def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """保存FAISS索引到指定路径"""
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # 保存索引
    faiss.write_index(index, file_path)
    print(f"索引已保存至：{file_path}")

def load_faiss_index(file_path: str) -> faiss.Index:
    """从文件加载FAISS索引"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"索引文件 {file_path} 不存在")
    
    return faiss.read_index(file_path)

# index = load_faiss_index("F:/faiss_index.faiss")
classify_df = pd.read_csv("./TerraPulse/models/base_M/inference_im2gps3ktest.csv")
def predict(query_image_path):
    """主函数，加载嵌入、查询图片并输出结果。"""
    # classify_df = classify_s2cell(query_image_path)
    # 加载嵌入数据
    global classify_df
    query_image_name = query_image_path.stem
    df = classify_df[classify_df['img_id'] == query_image_name]
    image_embeddings, text_list, image_id_list, cell_id_list = filter_embeddings(df, "hierarchy")

    # 构建 FAISS 索引
    index = build_faiss_index(image_embeddings)
    # save_faiss_index(index, "F:/faiss_index.faiss")
    # global index
    query_embedding = get_image_embedding(query_image_path)

    # 查找最相似的图片
    k = 5  # 设置相似图片数量
    positive_distances, positive_indices = search_most_similar_images(query_embedding, index, k)
    negative_distances, negative_indices = search_less_similar_images(query_embedding, index, k)
    # 显示结果
    # display_results(query_image_path, positive_indices, positive_distances, image_id_list, text_list, cell_id_list)
    # display_results(query_image_path, negative_indices, negative_distances, image_id_list, text_list, cell_id_list)

    most_similar_coords = get_coordinates_list(positive_indices, text_list)
    least_similar_coords = get_coordinates_list(negative_indices, text_list)
    prompt = f"""This is a query image. Estimate its geographic coordinates based on the reference data and visual features.

    - **Top {k} most similar images' coordinates**: {most_similar_coords}
    - **Bottom {k} least similar images' coordinates**: {least_similar_coords}

    ⚠️ The query image itself is NOT included in these sets. Do NOT use these coordinates directly. Instead, use them as references along with geographic clues in the image and general knowledge.
    **Output only the estimated coordinates on the first line in (latitude, longitude) format, e.g., (25.745593, -80.177536). No explanations. The answer cannot be empty.**"""
    print("*******-----------------********")
    return query_gpt_with_images_to_jsonl_append(query_image_path, prompt, Path("TerraPulse/output/queries.jsonl"))

if __name__ == "__main__":
    predict("TerraPulse/query.jpg")
