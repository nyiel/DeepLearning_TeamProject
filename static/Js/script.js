// DOM elements
const openCameraBtn = document.getElementById('openCameraBtn');
const cameraStream = document.getElementById('cameraStream');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const preview = document.getElementById('preview');
const imageInput = document.getElementById('imageUpload');
const uploadForm = document.getElementById('upload-form');
const loadingEl = document.getElementById('loading');
let stream = null;

// Open camera
openCameraBtn?.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        cameraStream.srcObject = stream;
        cameraStream.style.display = 'block';
        captureBtn.style.display = 'inline-block';
        openCameraBtn.style.display = 'none';
    } catch (err) {
        console.error(err);
        alert('Unable to access camera — please allow camera permission or upload an image.');
    }
});

// Capture from camera
captureBtn?.addEventListener('click', () => {
    if (!stream) return alert('Camera not started.');
    cameraCanvas.width = cameraStream.videoWidth;
    cameraCanvas.height = cameraStream.videoHeight;
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraStream, 0, 0, cameraCanvas.width, cameraCanvas.height);

    const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);
    preview.src = dataUrl;
    preview.style.display = 'block';

    // stop camera
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    cameraStream.style.display = 'none';
    captureBtn.style.display = 'none';
    openCameraBtn.style.display = 'inline-grid';
});

// Submit form
uploadForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    loadingEl.style.display = 'block';

    const fd = new FormData();

    if (imageInput.files && imageInput.files[0]) {
        fd.append('file', imageInput.files[0]);
    } else if (preview.src && preview.src.startsWith('data:image')) {
        const res = await fetch(preview.src);
        const blob = await res.blob();
        fd.append('file', blob, 'capture.jpg');
    } else {
        alert('Please upload or capture an image first.');
        loadingEl.style.display = 'none';
        return;
    }

    try {
        const response = await fetch('/predict', { method: 'POST', body: fd });
        if (response.redirected) {
            window.location.href = response.url;
            return;
        }
        const text = await response.text();
        document.open();
        document.write(text);
        document.close();
    } catch (err) {
        console.error(err);
        alert('Upload failed — try again');
    } finally {
        loadingEl.style.display = 'none';
    }
});

// Dark mode toggle
(function () {
    const key = 'mlc_theme';
    const stored = localStorage.getItem(key) || 'light';
    document.documentElement.setAttribute('data-theme', stored === 'dark' ? 'dark' : 'light');
})();

function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('mlc_theme', next);
}

window.toggleTheme = toggleTheme;
