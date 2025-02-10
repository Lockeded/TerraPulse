import os
import json
import torch
import clip
import pandas as pd
from PIL import Image

# 加载 CLIP 模型和预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 示例图片路径和经纬度坐标 CSV 文件路径
image_file_path = r"D:\mp16\downloads"  # 修改为实际图片路径
coordinate_file_path = r"mp16_places365.csv"  # 修改为实际 CSV 文件路径
cell_mapping_path = r"mp16_places365_mapping_h3.json"  # 修改为实际 JSON 文件路径

# 加载 CSV 文件
print(f"加载坐标数据：{coordinate_file_path}...")
coordinates_df = pd.read_csv(coordinate_file_path)
cell_mapping = json.load(open(cell_mapping_path, "r"))

# 创建一个字典，映射图片 ID 到经纬度坐标
coordinates_mapping = {}
for _, row in coordinates_df.iterrows():
    image_id = row['IMG_ID']  # 假设图片 ID 在 'IMG_ID' 列
    lat = row['LAT']  # 纬度
    lon = row['LON']  # 经度
    coordinates_mapping[image_id] = f"({lat}, {lon})"

print(f"坐标数据加载完成，包含 {len(coordinates_mapping)} 条记录。")

# 获取图片文件路径
image_files = [f for f in os.listdir(image_file_path)]
print(f"发现 {len(image_files)} 张图片。")

# 每批处理 1000 张图片
batch_size = 1000
image_embeddings_batch = []
text_batch = []
image_ids_batch = []  # 用于存储每个批次中图像的 ID
cell_ids_batch = []  # 用于存储每个批次中图像的 H3 单元 ID
index = 0

# 遍历图片文件夹中的所有图片
for idx, image_file in enumerate(image_files):
    # 获取图片路径
    image_path = os.path.join(image_file_path, image_file)

    # 从坐标映射中获取相应的地理坐标文本
    image_id = os.path.splitext(image_file)[0]  # 假设图片文件名不含扩展名就是 ID
    image_id = image_id.replace("_", "/") + ".jpg"  # 修正图片 ID
    coordinates = coordinates_mapping.get(image_id, None)

    if coordinates:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # 生成图片的特征嵌入
        with torch.no_grad():
            image_features = model.encode_image(image)

        # 将嵌入和 ID 添加到当前批次
        image_embeddings_batch.append(image_features.cpu())
        text_batch.append(coordinates)  # 直接将文本存储到批次中
        image_ids_batch.append(image_id)  # 存储当前图片的 ID
        cell_id = cell_mapping.get(image_id, None)
        cell_ids_batch.append(cell_id)
    else:
        print(f"图片 {image_file} 没有对应的坐标，跳过。")

    # 每处理完一批（1000 张图片），就保存一次
    if (idx + 1) % batch_size == 0 or (idx + 1) == len(image_files):
        print(f"处理到第 {idx + 1} 张图片，当前批次包含 {len(image_embeddings_batch)} 张图片的嵌入。")

        if image_embeddings_batch:  # 确保批次不为空
            # 合并当前批次的嵌入
            image_embeddings_batch = torch.cat(image_embeddings_batch, dim=0)

            # 保存当前批次的嵌入和 ID
            batch_filename = f"clip_embeddings_batch_{index}.pt"
            index += 1
            torch.save({
                "image": image_embeddings_batch,
                "text": text_batch,  # 直接存储文本
                "image_id": image_ids_batch,  # 保存当前批次的图像 ID
                "cell_id": cell_ids_batch  # 保存当前批次的 H3 单元 ID
            }, batch_filename)

            print(f"第 {(idx + 1)} 张图片处理完成，嵌入已保存为 {batch_filename}")

            # 清空批次数据，准备处理下一批
            image_embeddings_batch = []
            text_batch = []
            image_ids_batch = []  # 清空 ID 列表，准备下一批

        else:
            print(f"第 {(idx + 1)} 批处理时，未生成任何嵌入，跳过保存。")

print("所有嵌入处理完毕，逐批保存完成。")
