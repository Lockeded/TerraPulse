# app.py
from flask import Flask, request, jsonify, render_template
import os
from GeoLocRAG.GeoLocRAG import predict  # 导入你的预测模块
from GeoLocRAG.image_blur import blur
from hashlib import md5
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    if not file.filename.endswith('.jpg') and not file.filename.endswith('.png'):
        return jsonify({'error': 'Invalid file type'}), 400
    filename = md5(file.filename.encode()).hexdigest() + os.path.splitext(file.filename)[-1]
    
    # 保存上传的图片
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 调用你的预测模型
    try:
        response = predict(filepath).splitlines()
        lat, lng = response[0].replace("(","").replace(")","").split(',')  # 假设返回纬度和经度
        lat = float(lat)
        lng = float(lng)
        return jsonify({'lat': lat, 'lng': lng})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(use_reloader=True, debug=True)