import os
import torch
import faiss
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from openai import OpenAI
import base64
from .classify import classify
from pathlib import Path

with open("sk.txt", "r") as f:
        key = f.read().strip()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key = key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 全局设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 防止冲突报错


def load_embeddings_(embedding_dir: str, classify_df, target_p_key: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """加载所有嵌入文件，并提取图像嵌入、文本、图片 ID 和 H3 单元 ID，根据指定 p_key 过滤数据。"""
    target_p_key_dict = {"coarse": 0, "middle": 1, "fine": 2, "hierachy": 0}
    # 获取 classify_df 中指定 p_key 的所有 pred_class
    print(classify_df, target_p_key)
    target_pred_classes = classify_df[classify_df['p_key'] == target_p_key]['pred_class'].tolist()
    print(target_pred_classes)
    # 加载嵌入文件
    embedding_files = [
        os.path.join(embedding_dir, f) 
        for f in os.listdir(embedding_dir) 
        if f.startswith("clip_embeddings_batch_") and f.endswith(".pt")
    ]

    image_embeddings_list = []
    text_list = []
    image_id_list = []
    cell_id_list = []

    for file_path in embedding_files:
        embedding_data = torch.load(file_path, weights_only=True)

        # 过滤符合 target_pred_classes 的数据
        for idx, cell_id in enumerate(embedding_data["cell_id"]):
            # cell_id 是一个数组，有三个元素，分别对应不同的类别
            if(cell_id is None):
                continue
            if any(pred_class in target_pred_classes for pred_class in cell_id):
                image_embeddings_list.append(embedding_data["image"][idx].cpu().numpy())
                text_list.append(embedding_data["text"][idx])
                image_id_list.append(embedding_data["image_id"][idx])
                cell_id_list.append(cell_id)

    image_embeddings = np.concatenate(image_embeddings_list, axis=0)
    return image_embeddings, text_list, image_id_list, cell_id_list

def load_embeddings(embedding_dir: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """加载所有嵌入文件，并提取图像嵌入、文本、图片 ID 和 H3 单元 ID。"""
    embedding_files = [
        os.path.join(embedding_dir, f) 
        for f in os.listdir(embedding_dir) 
        if f.startswith("clip_embeddings_batch_") and f.endswith(".pt")
    ]

    image_embeddings_list = []
    text_list = []
    image_id_list = []
    cell_id_list = []

    for file_path in embedding_files:
        embedding_data = torch.load(file_path, weights_only=True)
        image_embeddings_list.append(embedding_data["image"].cpu().numpy())
        text_list.extend(embedding_data["text"])
        image_id_list.extend(embedding_data["image_id"])
        cell_id_list.extend(embedding_data["cell_id"])

    image_embeddings = np.concatenate(image_embeddings_list, axis=0)
    return image_embeddings, text_list, image_id_list, cell_id_list

def build_faiss_index(image_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """构建 FAISS 索引并添加归一化的图像嵌入。"""
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(image_embeddings.shape[1])  # 使用内积（余弦相似度）
    index.add(image_embeddings_norm)
    return index

def get_image_embedding(image_path: str) -> np.ndarray:
    """提取单张查询图片的嵌入特征。"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

def encode_image_2_base64(image_path: str) -> str:
    """将图片编码为 Base64 字符串。"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64

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
    base64_image = encode_image_2_base64(query_image_path)
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        seed=42,
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}, 
                },
                {"type": "text", "text": prompt},
            ],
        }
    ],
)
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


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
    """
    从给定的索引列表中提取坐标列表。
    参数:
    indices (np.ndarray): 包含索引的二维数组。
    text_list (List[str]): 包含所有文本的列表。
    返回:
    List[str]: 提取的坐标列表。
    """
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

def predict(query_image_path):
    """主函数，加载嵌入、查询图片并输出结果。"""
    embedding_dir = os.path.join(os.curdir, "TerraPulse")  # 修改为实际文件夹路径
    classify_df = classify_s2cell(query_image_path)
    # 加载嵌入数据
    # image_embeddings, text_list, image_id_list, cell_id_list = load_embeddings(embedding_dir, classify_df, target_p_key="hierarchy")
    image_embeddings, text_list, image_id_list, cell_id_list = load_embeddings(embedding_dir)

    # 构建 FAISS 索引
    index = build_faiss_index(image_embeddings)

    # 获取查询图片嵌入
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
    prompt_template = f"这是一张查询图片,假如你是一位图像地理定位方面的专家,请你结合你自己的知识,给出对这张图片地理坐标的猜测.这些是检索库中与其相似度最高的{k}张图片的地理坐标:{most_similar_coords},这些是与其相似度最低的{k}张的图片的地理坐标:{least_similar_coords},请注意查询图片并不在其中,所以不要直接使用这些坐标,而是作为参考,请你结合以上坐标信息,图像体现的地理位置特征以及你自己具备的知识,在推理后给出一个经过推理与计算的对查询图片地理坐标的猜测答案.你的答案必须在第一行以(LAT, LON)格式给出,不可为空,而后是你的推理思考过程."
    print("prompt: ",prompt_template)
    print("*******-----------------********")
    return query_gpt_with_images(query_image_path, prompt_template)

if __name__ == "__main__":
    predict(r"TerraPulse/resources/query.jpg")
