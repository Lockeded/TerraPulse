import os
import json
import clip
import torch
import pandas as pd
from PIL import Image

# 加载 CLIP 模型和预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 文件路径配置
image_file_path = r"D:\mp16\downloads"
coordinate_file_path = r"resources/mp16_places365.csv"
cell_mapping_path = r"resources/mp16_places365_mapping_h3.json"

# 加载数据
print("加载坐标数据...")
coordinates_df = pd.read_csv(coordinate_file_path)
cell_mapping = json.load(open(cell_mapping_path, "r"))

# 创建映射字典
coordinates_mapping = {}
for _, row in coordinates_df.iterrows():
    image_id = row['IMG_ID']
    coordinates_mapping[image_id] = f"({row['LAT']}, {row['LON']})"

print(f"坐标数据加载完成，共 {len(coordinates_mapping)} 条记录")

# 初始化存储容器
all_image_embeddings = []
all_texts = []
all_image_ids = []
all_cell_ids = []

# 遍历处理图片
image_files = [f for f in os.listdir(image_file_path)]
print(f"发现 {len(image_files)} 张图片，开始处理...")

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_file_path, image_file)
    image_id = os.path.splitext(image_file)[0].replace("_", "/") + ".jpg"
    
    if coordinates_mapping.get(image_id) is None:
        print(f"跳过无坐标图片：{image_id}")
        continue

    try:
        # 处理图片
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).cpu()
        
        # 收集数据
        all_image_embeddings.append(image_features)
        all_texts.append(coordinates_mapping[image_id])
        all_image_ids.append(image_id)
        all_cell_ids.append(cell_mapping.get(image_id, None))
        
    except Exception as e:
        print(f"处理图片 {image_file} 时出错：{str(e)}")
        continue

    # 进度提示
    if (idx + 1) % 100 == 0:
        print(f"已处理 {idx + 1}/{len(image_files)} 张图片")

# 合并所有嵌入
if all_image_embeddings:
    print("正在合并所有嵌入...")
    final_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    # 保存为单个文件
    output_filename = "clip_embeddings.pt"
    torch.save({
        "image_embeddings": final_embeddings,
        "texts": all_texts,
        "image_ids": all_image_ids,
        "cell_ids": all_cell_ids
    }, output_filename)
    
    print(f"所有嵌入已保存到 {output_filename}")
    print(f"总计保存 {final_embeddings.shape[0]} 个嵌入")
else:
    print("未生成任何有效嵌入，请检查输入数据")

print("处理完成")