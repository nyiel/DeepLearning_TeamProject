const resultDiv = document.getElementById('result');
resultDiv.style.display = 'none';

document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    resultDiv.style.display = 'block';
    document.getElementById('prediction').innerText = "Predicting...";
    document.getElementById('confidence').innerText = "";
    document.getElementById('gradcam').src = "";

    const formData = new FormData(this);
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
        alert(data.error);
        resultDiv.style.display = 'none';
    } else {
        document.getElementById('prediction').innerText = "Prediction: " + data.prediction;
        document.getElementById('confidence').innerText = "Confidence: " + data.confidence + "%";
        document.getElementById('gradcam').src = data.gradcam + '?t=' + new Date().getTime();
    }
});
