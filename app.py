from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from PIL import Image
from TerraPulse.query import predict
from TerraPulse.blur import blur
from hashlib import md5
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
BLUR_FOLDER = 'blurred'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BLUR_FOLDER'] = BLUR_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BLUR_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # 生成唯一文件名
    filename_hash = md5(file.filename.encode()).hexdigest()
    ext = os.path.splitext(file.filename)[-1]
    filename = filename_hash + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 获取图片信息
        img = Image.open(filepath)
        width, height = img.size
        file_size = os.path.getsize(filepath)
        img_format = img.format
        img_mode = img.mode
        img_hash = md5(img.tobytes()).hexdigest()

        # 调用地理预测模型
        response = predict(filepath).splitlines()
        lat, lng = response[0].replace("(", "").replace(")", "").split(',')
        reason = " ".join(response[1:]) if len(response) > 1 else "No additional information"

        result = {
            'lat': float(lat),
            'lng': float(lng),
            'message': reason,
            'filename': file.filename,
            'size': file_size,
            'width': width,
            'height': height,
            'format': img_format,
            'mode': img_mode,
            'hash': img_hash
        }
        return jsonify(result)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/blur', methods=['POST'])
def blur_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # 生成唯一文件名
    filename_hash = md5(file.filename.encode()).hexdigest()
    ext = os.path.splitext(file.filename)[-1]
    filename = filename_hash + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 处理模糊
    blurred_filename = f"blurred_{filename}"
    blurred_path = os.path.join(app.config['BLUR_FOLDER'], blurred_filename)

    try:
        blur(filepath, blurred_path)
       
        # 获取模糊后图片的信息
        img = Image.open(blurred_path)
        width, height = img.size
        file_size = os.path.getsize(blurred_path)
        img_format = img.format
        img_mode = img.mode
        img_hash = md5(img.tobytes()).hexdigest()

        return jsonify({
            'blurred_image_url': f'/blurred/{blurred_filename}',
            'filename': file.filename,
            'size': file_size,
            'width': width,
            'height': height,
            'format': img_format,
            'mode': img_mode,
            'hash': img_hash
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/blurred/<filename>')
def serve_blurred_image(filename):
    return send_from_directory(app.config['BLUR_FOLDER'], filename)

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True)

