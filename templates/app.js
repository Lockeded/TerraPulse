// 夜间模式切换
const themeToggle = document.getElementById('themeToggle');
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    themeToggle.querySelector('i').classList.toggle('fa-moon');
    themeToggle.querySelector('i').classList.toggle('fa-sun');
});

// 显示结果区域
function showResultSection(data) {
    const resultSection = document.querySelector('.result-section');
    resultSection.style.display = 'block';
    setTimeout(() => resultSection.style.opacity = '1', 10);
    const locationInfo = document.getElementById('locationInfo');
    const imageResolution = document.getElementById('imageResolution');
    const fileSize = document.getElementById('fileSize');
    const fileName = document.getElementById('fileName');
    const predictionReason = document.getElementById('predictionReason');
    console.log(data);
    // 填充位置信息
    locationInfo.innerHTML = `
        <p>纬度: ${data.lat.toFixed(6)}</p>
        <p>经度: ${data.lng.toFixed(6)}</p>
    `;

    // 填充图像信息
    imageResolution.innerHTML = `${data.width} x ${data.height}`;
    fileSize.innerHTML = `${(data.size / 1024).toFixed(2)} KB`;
    fileName.innerHTML = data.filename;

    // 显示预测理由
    predictionReason.textContent = data.message || '未提供预测理由';

    resultSection.style.display = 'block';
}

// 图像模糊处理
document.getElementById('blurButton').addEventListener('click', async () => {
    const file = document.getElementById('imageInput').files[0];
    if (!file) {
        showError('请先上传图片');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/blur', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            // 更新图像显示
            const imgPreview = document.getElementById('imagePreview');
            if (imgPreview) {
                imgPreview.src = data.blurred_image_url;
            }
            // 更新按钮状态
            const blurButton = document.getElementById('blurButton');
            blurButton.disabled = true;
            blurButton.classList.remove('btn-outline-light');
            blurButton.classList.add('btn-success');
            blurButton.innerHTML = `<i class="fas fa-check-circle me-2"></i>已模糊处理` ;
            showSuccess('图像已成功模糊处理');
            const predictionReason = document.getElementById('predictionReason');
            predictionReason.textContent = '图像中的地理位置信息已模糊处理';
            predictionReason.style.backgroundColor = 'rgba(25, 135, 84, 0.1)';
        }
    } catch (error) {
        showError('请求失败: ' + error.message);
    }
});

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'position-fixed bottom-0 end-0 m-3 toast align-items-center text-white bg-success border-0';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    document.body.appendChild(toast);
    new bootstrap.Toast(toast).show();
}

// 初始化地图
const map = L.map('map', {
    layers: [L.tileLayer('https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
        attribution: '© 高德地图'
    })]
}).setView([30.5, 114.3], 3);

// 自定义地图样式
map.attributionControl.setPrefix('');

// 文件上传处理
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            // 清除旧标记
            map.eachLayer(layer => layer instanceof L.Marker && map.removeLayer(layer));
            showResultSection(data);
            console.log(data);
            // 创建自定义标记
            const marker = L.marker([data.lat, data.lng], {
                icon: L.divIcon({ className: 'result-marker' })
            }).addTo(map);

            // 平滑定位动画
            map.flyTo([data.lat, data.lng], 15, {
                animate: true,
                duration: 1.5
            });

            // 信息弹窗
            marker.bindPopup(`
                <div class="p-2">
                    <h6 class="mb-2">📍 预测位置</h6>
                    <table class="table table-sm">
                        <tr><td>纬度</td><td>${data.lat.toFixed(6)}</td></tr>
                        <tr><td>经度</td><td>${data.lng.toFixed(6)}</td></tr>
                    </table>
                </div>
            `).openPopup();
        }
    } catch (error) {
        showError('请求失败: ' + error.message);
    }
});

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'position-fixed bottom-0 end-0 m-3 toast align-items-center text-white bg-danger border-0';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    document.body.appendChild(toast);
    new bootstrap.Toast(toast).show();
}