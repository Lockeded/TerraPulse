import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 防止冲突报错
import cv2
import easyocr
import numpy as np
from pathlib import Path

# ==== 常量参数 ====
MODEL_PATH = 'C:/Users/10139/Documents/SelfLearning/ImageConan/TerraPulse/models/base_M/face_detection_yunet_2022mar.onnx'
LANGUAGE = ['ch_sim', 'en']
CONFIDENCE = 0.001
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
KERNEL_SIZE = (15, 15)
BACKEND = cv2.dnn.DNN_BACKEND_DEFAULT
TARGET = cv2.dnn.DNN_TARGET_CPU

def blur_faces_and_text(image_path: str) -> str:
    """
    检测并模糊图像中的人脸和文字区域。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像 {image_path}")
        return

    h, w = image.shape[:2]

    # === 人脸检测（YuNet）===
    yunet = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config='',
        input_size=(w, h),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
        backend_id=BACKEND,
        target_id=TARGET
    )

    yunet.setInputSize((w, h))
    _, faces = yunet.detect(image)
    face_rects = []

    if faces is not None:
        for face in faces:
            x, y, width, height = face[:4].astype(int)
            face_rects.append((x, y, width, height))

    # === 文本检测（EasyOCR）===
    reader = easyocr.Reader(LANGUAGE, gpu=True)
    result = reader.readtext(image)
    text_rects = []
    for r in result:
        if r[2] >= CONFIDENCE:
            points = r[0]
            x = int(min(p[0] for p in points))
            y = int(min(p[1] for p in points))
            w = int(max(p[0] for p in points) - x)
            h = int(max(p[1] for p in points) - y)
            text_rects.append((x, y, w, h))

    # === 模糊区域处理 ===
    all_rects = face_rects + text_rects
    for rect in all_rects:
        x, y, w, h = rect
        if w > 0 and h > 0:
            roi = image[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, KERNEL_SIZE, 0)
            image[y:y+h, x:x+w] = blurred_roi

    # === 保存图像 ===
    output_dir = os.path.join(os.path.dirname(image_path), "blur_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"处理后的图像已保存至 {output_path}")
    return output_path

def batch_blur_faces_and_text(folder_path: str):
    """
    批量处理文件夹内所有图片，进行人脸和文字的检测与模糊处理。
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹不存在: {folder_path}")

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

    total_images = len(image_files)
    print(f"开始处理文件夹 {folder_path} 中的 {total_images} 张图片...")

    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        print(f"正在处理第 {i}/{total_images} 张图片: {image_file}")
        try:
            blur_faces_and_text(image_path)
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")

    print("批量处理完成。")

if __name__ == "__main__":
    folder_path = "TerraPulse/images"
    batch_blur_faces_and_text(folder_path)