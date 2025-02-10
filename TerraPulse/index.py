import os
import torch
import faiss
import numpy as np

# FAISS 索引初始化函数
def create_faiss_index(image_embeddings, text_embeddings):
    # 获取图像嵌入的维度（假设图像和文本嵌入的维度相同）
    dim = image_embeddings.shape[1]

    # 初始化 FAISS 索引
    index = faiss.IndexFlatL2(dim)  # 使用 L2 距离（可以根据需要选择不同的索引类型）

    # 将图像嵌入加入到索引中
    index.add(image_embeddings.astype(np.float32))  # 转换为 float32 类型
    print(f"图像嵌入已加入索引，总数: {index.ntotal}")

    return index

# 加载 .pt 文件并提取嵌入
def load_embeddings_from_pt(file_path):
    data = torch.load(file_path)
    image_embeddings = data["image"].cpu().numpy()  # 转换为 NumPy 数组
    text_embeddings = data["text"]
    return image_embeddings, text_embeddings

# 主程序
def main():
    # 目录路径
    embeddings_dir = os.curdir  # 修改为实际存储 .pt 文件的目录路径
    faiss_index_file = r"faiss_index_file.index"  # FAISS 索引文件保存路径

    # 创建一个空的列表来存储所有嵌入
    all_image_embeddings = []
    all_text_embeddings = []

    # 遍历目录中的所有 .pt 文件
    for pt_file in os.listdir(embeddings_dir):
        if pt_file.endswith(".pt"):
            file_path = os.path.join(embeddings_dir, pt_file)
            print(f"加载 {file_path}...")
            
            # 加载嵌入
            image_embeddings, text_embeddings = load_embeddings_from_pt(file_path)

            # 添加到总列表中
            all_image_embeddings.append(image_embeddings)
            all_text_embeddings.append(text_embeddings)

    # 将所有的嵌入合并为一个大数组
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)

    # 创建 FAISS 索引
    print("创建 FAISS 索引...")
    index = create_faiss_index(all_image_embeddings, all_text_embeddings)

    # 保存 FAISS 索引
    faiss.write_index(index, faiss_index_file)
    print(f"FAISS 索引已保存到 {faiss_index_file}")

if __name__ == "__main__":
    main()
