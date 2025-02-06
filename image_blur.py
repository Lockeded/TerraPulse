import argparse
from pathlib import Path
import os
import easyocr
import cv2

VERSION = "1.2.4"

def blur_text_regions(image_path, reader, confidence, preview=False):
    """
    Detect and blur text regions in the given image.

    Args:
        image_path (str): Path to the image file.
        reader (easyocr.Reader): EasyOCR reader instance.
        confidence (float): Confidence threshold for OCR detection.
        preview (bool): Whether to preview the image with detected regions.

    Returns:
        str: Path to the processed image file.
    """
    result = reader.readtext(image_path)

    # Read the image
    img_rect = cv2.imread(image_path)
    img_temp = cv2.imread(image_path)
    h, w, c = img_temp.shape

    # Fill temp image with black
    img_temp = cv2.rectangle(img_temp, [0, 0], [w, h], (0, 0, 0), -1)
    img_inpaint = cv2.imread(image_path)

    preview_rect = cv2.imread(image_path)

    for r in result:
        # If the OCR text confidence is above the threshold
        if r[2] >= confidence:
            bottom_left = tuple(int(x) for x in tuple(r[0][0]))
            top_right = tuple(int(x) for x in tuple(r[0][2]))

            # Draw a rectangle and mask the region
            img_rect = cv2.rectangle(img_rect, bottom_left, top_right, (0, 255, 0), 3)
            img_temp = cv2.rectangle(img_temp, bottom_left, top_right, (255, 255, 255), -1)

            # Convert temp image to black and white for mask
            mask = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

            # Perform inpainting
            img_inpaint = cv2.inpaint(img_inpaint, mask, 3, cv2.INPAINT_TELEA)

            # Draw rectangle and confidence level for preview
            preview_rect = cv2.rectangle(img_rect, bottom_left, top_right, (0, 255, 0), 3)
            cv2.putText(preview_rect, str(round(r[2], 2)), bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)



    # Save the processed image
    output_path = image_path.replace(".png", "").replace(".jpg", "") + "_ocr.png"
    cv2.imwrite(output_path, img_inpaint)

    return output_path

PATH = "image.png"
LANGUAGE = 'ch_sim'
CONFIDENCE = 0.001

# Check if file path exists
target_dir = Path(PATH)

if not target_dir.exists():
    print("The target file/directory doesn't exist")
    raise SystemExit(1)

# Get list of images to be OCR'd
if os.path.isdir(PATH):
    files = os.listdir(PATH)
    images = [file for file in files if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
else:
    images = [PATH]

# OCR the images
reader = easyocr.Reader([LANGUAGE])
for image in images:
    output_path = blur_text_regions(image, reader, CONFIDENCE)
    print(f"Processed image saved at: {output_path}")
