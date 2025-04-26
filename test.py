'''
    用于对基线数据集im2gps3k进行测试
    1. 加载im2gps3k的坐标数据
    2. 加载模型
    3. 对im2gps3k的图片进行预测
    4. 计算预测坐标和真实坐标之间的距离，并返回各个距离范围内的准确率。
'''
from TerraPulse.classify import classify
# from TerraPulse.query import *
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic
import json
from tqdm import tqdm
def calculate_accuracy_with_geopy(result_df: pd.DataFrame):
    """
    使用 geopy 库计算预测坐标和真实坐标之间的距离，并返回各个距离范围内的准确率。
    
    参数:
        result_df (pd.DataFrame): 包含 'Predicted_LAT', 'Predicted_LON', 'True_LAT', 'True_LON' 的 DataFrame。
        
    返回:
        dict: 各个距离范围内的准确率。
    """
    thresholds = [1, 25, 200, 750, 2500]  # 距离阈值（单位：km）
    
    # 初始化计数器
    accuracy = {threshold: 0 for threshold in thresholds}
    total_count = len(result_df)
    
    for _, row in result_df.iterrows():
        true_lat = row['True_LAT']
        true_lon = row['True_LON']
        predicted_lat = row['Predicted_LAT']
        predicted_lon = row['Predicted_LON']
        
        # 使用 geopy 计算距离
        true_coords = (true_lat, true_lon)
        predicted_coords = (predicted_lat, predicted_lon)
        if(predicted_coords == (0.0, 0.0)):
            continue
        distance = geodesic(true_coords, predicted_coords).kilometers
        
        # 根据距离判断准确性
        for threshold in thresholds:
            if distance <= threshold:
                accuracy[threshold] += 1
    
    # 计算准确率
    # 假设 accuracy 是一个字典，存储每个阈值的准确计数
    accuracy_percentage = {
        threshold: f"{(count / total_count) * 100:.4f}%"  # 保留小数点后两位并加上百分号
        for threshold, count in accuracy.items()
    }
        
    return accuracy_percentage


def load_image_coords(url_csv: Path):
    df = pd.read_csv(url_csv)
    # 提取需要的列
    coords_df = df[['IMG_ID', 'LAT', 'LON']]
    return coords_df

def load_response(path):
    with open(path, "r", encoding="utf-8") as f:
        response = [json.loads(line.strip()) for line in f]
    return response

def load_inference_result(path):
    df = pd.read_csv(path)
    return df

def test_im2gps3k(checkpoint: Path, test_valset_path: Path, test_dataset_path: Path, use_gpu: bool):
    # 加载图像坐标数据
    coords_df = load_image_coords(test_valset_path)

    # 用于存储预测结果的列表
    predictions = []

    # 循环处理每张图像
    image_files = [
        f for f in test_dataset_path.iterdir()
        if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
    ]
    
    response_list = load_response("TerraPulse/output/hunyuan_result.jsonl")
    inference_df = load_inference_result("TerraPulse/models/base_M/inference_im2gps3ktest.csv")
    count = 0
    # for image_path in tqdm(image_files, desc="Processing images"): 
    #     if image_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:  # 确保只处理图像文件
    #         predict(image_path)
    #         continue
    for row in response_list:
            # img_id = row['custom_id']
            # response = row["response"]["body"]["choices"][0]["message"]["content"]
        for img_id, response in row.items():
            try:
                predicted_lat, predicted_lon = response.replace("(", "").replace(")", "").split(',')
                predicted_lat = float(predicted_lat)
                predicted_lon = float(predicted_lon)
                if(predicted_lat == 0.0 and predicted_lon == 0.0):
                    mask = (inference_df['p_key'] == "hierarchy") & (inference_df['img_id'] == img_id.replace(".jpg", ""))
                    predicted_lat = float(inference_df.loc[mask, 'pred_lat'].values[0])
                    predicted_lon = float(inference_df.loc[mask, 'pred_lng'].values[0])
                    print(f"Use inference result {predicted_lat}, {predicted_lon}")
            except:
                print(f"Warning: Dangerous response {response} for image {img_id}")
                # predicted_lat = 0.0
                # predicted_lon = 0.0
                mask = (inference_df['p_key'] == "hierarchy") & (inference_df['img_id'] == img_id.replace(".jpg", ""))
                predicted_lat = float(inference_df.loc[mask, 'pred_lat'].values[0])
                predicted_lon = float(inference_df.loc[mask, 'pred_lng'].values[0])
                print(f"Use inference result {predicted_lat}, {predicted_lon}")
            
            # # 提取图像 ID，这里假设图像文件名即为 IMG_ID
            # img_id = image_path.stem + image_path.suffix

            # 从 coords_df 中获取对应的真实经纬度（LAT, LON）
            true_coords = coords_df[coords_df['IMG_ID'] == img_id]
            if not true_coords.empty:
                true_lat = true_coords['LAT'].values[0]
                true_lon = true_coords['LON'].values[0]
            else:
                print(f"Warning: No coordinates found for image {img_id}")
                continue

            # 存储图像的预测结果及真实标签
            predictions.append({
                'IMG_ID': img_id,
                'Predicted_LAT': predicted_lat,
                'Predicted_LON': predicted_lon,
                'True_LAT': true_lat,
                'True_LON': true_lon
            })

    # 将预测结果存储到 DataFrame 中
    result_df = pd.DataFrame(predictions)

    # 打印或保存结果
    print(result_df)
    result_df.to_csv("./TerraPulse/output/result_df.csv", index=False)
    
    accuracy_df = calculate_accuracy_with_geopy(result_df)
    print(accuracy_df)
    json.dump(accuracy_df, open("accuracy.json", "w"))
    
if __name__ == "__main__":
    checkpoint_path = Path("TerraPulse/models/base_M/epoch=014-val_loss=18.4833.ckpt")
    test_valset_path = Path("TerraPulse/resources/images/im2gps3k_places365.csv")
    test_dataset_path = Path("TerraPulse/resources/images/im2gps3k")
    use_gpu = False
    test_im2gps3k(checkpoint_path, test_valset_path, test_dataset_path, use_gpu)
