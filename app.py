# app.py
from flask import Flask, request, jsonify, render_template
import os
from GeoLocRAG import GeoLocRAG  # 导入你的预测模块

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
    
    # 保存上传的图片
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # 调用你的预测模型
    try:
        response = GeoLocRAG(filepath).splitlines()
        lat, lng = response[0].replace("(","").replace(")","").split(',')  # 假设返回纬度和经度
        lat = float(lat)
        lng = float(lng)
        return jsonify({'lat': lat, 'lng': lng})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False)