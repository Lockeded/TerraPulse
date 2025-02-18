// app.js
document.addEventListener('DOMContentLoaded', () => {
    // 全局状态管理
    const state = {
        currentLocation: null,
        uploadedFile: null,
        isDarkMode: false
    };

    // 地图初始化
    const map = L.map('map', {
        layers: [L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        })]
    }).setView([35, 105], 4);

    // DOM 元素引用
    const elements = {
        themeToggle: document.getElementById('themeToggle'),
        imageInput: document.getElementById('imageInput'),
        uploadLabel: document.querySelector('.upload-label'),
        locationInfo: document.getElementById('locationInfo'),
        predictionReason: document.getElementById('predictionReason'),
        imageResolution: document.getElementById('imageResolution'),
        fileSize: document.getElementById('fileSize'),
        fileName: document.getElementById('fileName'),
        blurButton: document.getElementById('blurButton')
    };

    // 主题切换功能
    elements.themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        state.isDarkMode = !state.isDarkMode;
        elements.themeToggle.querySelector('i').classList.toggle('fa-moon');
        elements.themeToggle.querySelector('i').classList.toggle('fa-sun');
    });

    // 文件上传处理
    elements.imageInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            // 更新上传状态
            elements.uploadLabel.innerHTML = `<i class="fas fa-spinner fa-spin"></i> 分析中...`;
            
            // 创建预览
            createImagePreview(file);
            
            // 发送请求
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.error) throw new Error(data.error);

            // 更新状态
            state.currentLocation = { lat: data.lat, lng: data.lng };
            state.uploadedFile = file;

            // 更新界面
            updateResultDisplay(data);
            updateMapMarker(data);
        } catch (error) {
            showError(`分析失败: ${error.message}`);
        } finally {
            elements.uploadLabel.innerHTML = `<i class="fas fa-cloud-upload-alt"></i> 点击上传或拖放文件`;
        }
    });

    // 图片预览功能
    function createImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewContainer = document.createElement('div');
            previewContainer.className = 'image-preview fade-in';
            previewContainer.innerHTML = `
                <img src="${e.target.result}" class="preview-image" alt="图片预览">
                <div class="preview-overlay">
                    <i class="fas fa-search-plus"></i>
                </div>
            `;
            document.querySelector('.upload-section').appendChild(previewContainer);
        };
        reader.readAsDataURL(file);
    }

    // 更新结果展示
    function updateResultDisplay(data) {
        elements.locationInfo.innerHTML = `
            <div>纬度: ${data.lat.toFixed(6)}</div>
            <div>经度: ${data.lng.toFixed(6)}</div>
        `;
        elements.imageResolution.textContent = `${data.width} × ${data.height}`;
        elements.fileSize.textContent = `${(data.size / 1024).toFixed(2)} KB`;
        elements.fileName.textContent = data.filename;
        elements.predictionReason.textContent = data.message || '未提供预测理由';
    }

    // 更新地图标记
    function updateMapMarker(data) {
        // 清除旧标记
        map.eachLayer(layer => layer instanceof L.Marker && map.removeLayer(layer));

        // 添加新标记
        const marker = L.marker([data.lat, data.lng], {
            icon: L.divIcon({ className: 'result-marker' })
        }).addTo(map);

        // 设置地图视图
        map.flyTo([data.lat, data.lng], 15, {
            animate: true,
            duration: 1.5
        });

        // 添加弹出信息
        marker.bindPopup(`
            <div class="map-popup">
                <h6>📍 预测位置</h6>
                <div class="map-coordinates">
                    <div>纬度: ${data.lat.toFixed(6)}</div>
                    <div>经度: ${data.lng.toFixed(6)}</div>
                </div>
            </div>
        `).openPopup();
    }

    // 模糊处理功能
    elements.blurButton.addEventListener('click', async () => {
        if (!state.uploadedFile) {
            showError('请先上传图片');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', state.uploadedFile);
            
            const response = await fetch('/blur', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.error) throw new Error(data.error);

            // 更新界面状态
            elements.blurButton.disabled = true;
            elements.blurButton.classList.add('processed');
            elements.blurButton.innerHTML = `<i class="fas fa-check-circle"></i> 已模糊处理`;
            elements.predictionReason.textContent = '地理位置信息已模糊处理';
            showSuccess('图像处理成功');
        } catch (error) {
            showError(`模糊处理失败: ${error.message}`);
        }
    });

    // 工具函数
    function showError(message) {
        const toast = document.createElement('div');
        toast.className = 'toast error-toast';
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-times-circle"></i>
                <span>${message}</span>
            </div>
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }

    function showSuccess(message) {
        const toast = document.createElement('div');
        toast.className = 'toast success-toast';
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-check-circle"></i>
                <span>${message}</span>
            </div>
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }
});