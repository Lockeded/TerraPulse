import os
import json
import clip
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.eval()  # **启用评估模式**

# **优化参数**
BATCH_SIZE = 64  # 根据GPU显存调整
NUM_WORKERS = cpu_count() // 2  # **使用半数CPU核心**

# 路径配置
image_dir = r"D:\mp16\mp16\r13_local_data\mp16\mp16_rgb_images"
# image_dir = r"D:\mp16\part"
coord_csv = r"resources/mp16_places365.csv"
cell_mapping_path = r"resources/mp16_places365_mapping_h3.json"
output_path = r"F:/clip_embeddings.pt"
coord_cache_path = r"resources/coord_mapping.bin"  # **新增缓存文件路径**

def load_or_create_coord_mapping():
    """**智能加载/创建坐标映射**"""
    if os.path.exists(coord_cache_path):
        try:
            # 使用torch加载二进制文件（关键修正）
            print("从二进制缓存加载坐标映射...")
            return torch.load(coord_cache_path, weights_only=True)
        except Exception as e:
            print(f"缓存加载失败: {e}, 重新生成...")
    
    print("从CSV生成坐标映射...")
    coord_df = pd.read_csv(coord_csv)
    mapping = {row['IMG_ID']: f"({row['LAT']}, {row['LON']})" 
              for _, row in coord_df.iterrows()}
    
    # **异步保存避免阻塞主线程**
    torch.save(mapping, coord_cache_path)  # 使用torch保存二进制格式
    return mapping
# **自定义数据集类**
class ImageDataset(Dataset):
    def __init__(self, valid_files, coord_mapping):
        self.valid_files = valid_files
        self.coord_mapping = coord_mapping

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        rel_path = self.valid_files[idx]
        image_path = os.path.join(image_dir, rel_path)
        image_id = rel_path.replace(os.path.sep, '/')
        base, ext = os.path.splitext(image_id)
        image_id = base + ext.lower()
        
        try:
            image = Image.open(image_path).convert("RGB")
            return preprocess(image), image_id
        except:
            return None, None  # **返回空值处理异常**

# **批量处理函数**
def collate_fn(batch):
    images = []
    ids = []
    for item in batch:
        if item[0] is not None:
            images.append(item[0])
            ids.append(item[1])
    return torch.stack(images), ids  # **自动堆叠张量**

def main():
    # 加载元数据
    coord_mapping = load_or_create_coord_mapping()  # **替换原有加载逻辑**
    cell_mapping = json.load(open(cell_mapping_path, "r"))

    # **构建数据集**
    valid_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), image_dir)
                valid_files.append(rel_path)
    
    dataset = ImageDataset(valid_files, coord_mapping)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True  # **加速数据到GPU的传输**
    )

    # 初始化存储
    results = {
        "image": [],
        "text": [],
        "image_id": [],
        "cell_id": []
    }

    # **批量处理**
    with torch.no_grad():
        for batch, batch_ids in tqdm(dataloader, desc="Processing"):
            if not batch_ids:
                continue
            
            # **GPU批量推理**
            batch = batch.to(device)
            features = model.encode_image(batch).cpu()
            
            # **收集结果**
            valid_indices = [i for i, img_id in enumerate(batch_ids) 
                           if img_id in coord_mapping]
            
            results["image"].append(features[valid_indices])
            results["text"].extend([coord_mapping[img_id] 
                                  for img_id in batch_ids 
                                  if img_id in coord_mapping])
            results["image_id"].extend([img_id for img_id in batch_ids 
                                      if img_id in coord_mapping])
            results["cell_id"].extend([cell_mapping.get(img_id, None) 
                                     for img_id in batch_ids 
                                     if img_id in coord_mapping])

    # 合并结果
    if results["image"]:
        final_dict = {
            "image": torch.cat(results["image"], dim=0),
            "text": results["text"],
            "image_id": results["image_id"],
            "cell_id": results["cell_id"]
        }
        torch.save(final_dict, output_path)
        print(f"保存完成，总样本数: {len(final_dict['image_id'])}")
    else:
        print("无有效数据")

if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn')  # **解决多进程问题**
    main()