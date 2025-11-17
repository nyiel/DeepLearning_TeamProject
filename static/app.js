const resultDiv = document.getElementById('result');
resultDiv.style.display = 'none';

document
  .getElementById('upload-form')
  .addEventListener('submit', async function (e) {
    e.preventDefault();

    resultDiv.style.display = 'block';
    document.getElementById('prediction').innerText = "Predicting...";
    document.getElementById('confidence').innerText = "";
    document.getElementById('gradcam').src = "";

    const formData = new FormData(this);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (data.error) {
        alert(data.error);
        resultDiv.style.display = 'none';
        return;
      }

      document.getElementById('prediction').innerText =
        "Prediction: " + data.prediction;

      document.getElementById('confidence').innerText =
        "Confidence: " + data.confidence + "%";

      // Avoid caching Grad-CAM image
      document.getElementById('gradcam').src =
        data.gradcam + '?t=' + new Date().getTime();

    } catch (err) {
      console.error("Prediction request failed:", err);
      alert("Error: Unable to process prediction.");
      resultDiv.style.display = 'none';
    }
  });

