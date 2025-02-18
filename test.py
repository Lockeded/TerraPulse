from TerraPulse.classify import classify
from TerraPulse.query import *
import pandas as pd
from geopy.distance import geodesic

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

def test_im2gps3k(checkpoint: Path, test_valset_path: Path, test_dataset_path: Path, use_gpu: bool):
    # 加载图像坐标数据
    coords_df = load_image_coords(test_valset_path)

    # 假设你有一个模型函数用于进行预测
    # model = load_model(checkpoint)  # 根据需要加载模型

    # 用于存储预测结果的列表
    predictions = []

    # 循环处理每张图像
    for image_path in test_dataset_path.iterdir():
        if image_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:  # 确保只处理图像文件

            # 示例: 假设你的预测结果是经纬度 (latitude, longitude)
            predicted_lat, predicted_lon = 0.0 , 0.0  # 使用实际模型的输出替代

            # 提取图像 ID，这里假设图像文件名即为 IMG_ID
            img_id = image_path.stem

            # 从 coords_df 中获取对应的真实经纬度（LAT, LON）
            true_coords = coords_df[coords_df['IMG_ID'] == img_id+".jpg"]
            if not true_coords.empty:
                predicted_lat, predicted_lon = 32.325436 , -64.764404  # 使用实际模型的输出替代
                true_lat = true_coords['LAT'].values[0]
                true_lon = true_coords['LON'].values[0]
            else:
                print(f"Warning: No coordinates found for image {img_id}")
                pass

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
    result_df.to_csv("result_df.csv", index=False)
    
    accuracy_df = calculate_accuracy_with_geopy(result_df)
    print(accuracy_df)
    accuracy_df.to_csv("accuracy_df.csv", index=False)
    
if __name__ == "__main__":
    checkpoint_path = Path("TerraPulse/models/base_M/epoch=014-val_loss=18.4833.ckpt")
    test_valset_path = Path("TerraPulse/resources/images/im2gps3k_places365.csv")
    test_dataset_path = Path("TerraPulse/resources/images/im2gps3k")
    use_gpu = False
    test_im2gps3k(checkpoint_path, test_valset_path, test_dataset_path, use_gpu)
