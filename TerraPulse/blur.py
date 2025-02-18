import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 防止冲突报错
import cv2
import easyocr
from pathlib import Path

def blur(image_path: str, language: list = ['ch_sim','en'], confidence: float = 0.001) -> str:
    """
    Detect and blur text regions in the given image.

    Args:
        image_path (str): Path to the image file.
        language (str): Language for OCR detection.
        confidence (float): Confidence threshold for OCR detection.

    Returns:
        str: Path to the processed image file.
    """
    reader = easyocr.Reader(language, gpu=True)
    result = reader.readtext(image_path)

    # Read the image
    img_temp = cv2.imread(image_path)
    h, w, _ = img_temp.shape

    # Create mask image
    img_temp = cv2.rectangle(img_temp, [0, 0], [w, h], (0, 0, 0), -1)
    img_inpaint = cv2.imread(image_path)

    for r in result:
        if r[2] >= confidence:
            bottom_left = tuple(int(x) for x in tuple(r[0][0]))
            top_right = tuple(int(x) for x in tuple(r[0][2]))
            img_temp = cv2.rectangle(img_temp, bottom_left, top_right, (255, 255, 255), -1)

    # Convert temp image to black and white for mask
    mask = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    # Perform inpainting
    img_inpaint = cv2.inpaint(img_inpaint, mask, 3, cv2.INPAINT_TELEA)

    # Save the processed image
    output_dir = os.path.join(os.path.dirname(image_path), "blur_results")
    os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img_inpaint)
    
    return output_path

if __name__ == "__main__":
    
    image_folder_path = f"resources/images/im2gps3k"
    print(f"Processing images in folder: {image_folder_path}")
    for image_path in Path(image_folder_path).iterdir():
        if image_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            print(f"Processing image: {image_path}")
            output_path = blur(str(image_path))

    
