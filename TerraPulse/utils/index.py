import os
import faiss
import numpy as np
import torch
from typing import List, Tuple, Dict

# 加载嵌入数据
embedding_path = r"F:/clip_embeddings.pt"
embedding_data = torch.load(embedding_path, weights_only=True)

def load_embeddings() -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """加载嵌入数据。"""
    image_embeddings_list = embedding_data["image"].cpu().numpy()
    text_list = embedding_data["text"]
    image_id_list = embedding_data["image_id"]
    cell_id_list = embedding_data["cell_id"]
    return image_embeddings_list, text_list, image_id_list, cell_id_list

def filter_embeddings_by_cell(cell_id: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """根据特定cell_id过滤嵌入数据"""
    filtered_embeddings, filtered_texts, filtered_image_ids, filtered_cell_ids = [], [], [], []
    for i in range(len(embedding_data["cell_id"])):
        if str(embedding_data["cell_id"][i][2]) == cell_id:  # 选择 "hierarchy" 层级
            filtered_embeddings.append(embedding_data["image"][i].numpy())
            filtered_texts.append(embedding_data["text"][i])
            filtered_image_ids.append(embedding_data["image_id"][i])
            filtered_cell_ids.append(embedding_data["cell_id"][i])
    
    final_embeddings = np.stack(filtered_embeddings) if filtered_embeddings else np.array([])
    return final_embeddings, filtered_texts, filtered_image_ids, [cid[2] for cid in filtered_cell_ids]

def build_faiss_index(image_embeddings: np.ndarray) -> faiss.Index:
    """构建FAISS索引"""
    if image_embeddings.size == 0:
        return None
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(image_embeddings_norm.shape[1])
    index.add(image_embeddings_norm)
    return index

def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """保存FAISS索引到指定路径"""
    if index is None or not index.is_trained:
        print(f"索引 {file_path} 为空或未训练，跳过保存。")
        return
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    faiss.write_index(index, file_path)
    print(f"索引已保存至：{file_path}")

def generate_indices_for_all_cells():
    """为所有唯一cell_id生成FAISS索引"""
    _, _, _, cell_id_list = load_embeddings()
    unique_cell_ids = set([str(cid[2]) for cid in cell_id_list if len(cid) > 2])

    for cell_id in unique_cell_ids:
        print(f"正在为cell_id: {cell_id} 构建索引...")
        image_embeddings, _, _, _ = filter_embeddings_by_cell(cell_id)
        index = build_faiss_index(image_embeddings)
        if index is not None:
            save_faiss_index(index, f"F:/faiss_indices/faiss_index_cell_{cell_id}.faiss")

if __name__ == "__main__":
    generate_indices_for_all_cells()
