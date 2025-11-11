document.addEventListener("DOMContentLoaded", () => {
  const uploadInput = document.getElementById("imageUpload");
  const previewImg = document.getElementById("preview");
  const uploadForm = document.getElementById("upload-form");
  const loadingText = document.getElementById("loading");

  const openCameraBtn = document.getElementById("openCameraBtn");
  const cameraStream = document.getElementById("cameraStream");
  const captureBtn = document.getElementById("captureBtn");
  const cameraCanvas = document.getElementById("cameraCanvas");

  let stream;

  // 📸 Open webcam
  openCameraBtn.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cameraStream.srcObject = stream;
      cameraStream.style.display = "block";
      captureBtn.style.display = "inline-block";
      previewImg.style.display = "none";
    } catch (err) {
      alert("Unable to access the camera: " + err.message);
    }
  });

  // 🖼️ Capture image from video
  captureBtn.addEventListener("click", () => {
    const ctx = cameraCanvas.getContext("2d");
    cameraCanvas.width = cameraStream.videoWidth;
    cameraCanvas.height = cameraStream.videoHeight;
    ctx.drawImage(cameraStream, 0, 0);

    cameraCanvas.toBlob((blob) => {
      const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
      showPreview(file);
      uploadInput.files = new DataTransfer().files;
      stopCamera();
    }, "image/jpeg");
  });

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      cameraStream.style.display = "none";
      captureBtn.style.display = "none";
    }
  }

  // 🖼️ Show preview
  function showPreview(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  uploadInput.addEventListener("change", (e) => showPreview(e.target.files[0]));

  // 🚀 Predict leaf
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = uploadInput.files[0];
    if (!file) return alert("Please upload or capture an image first!");

    const formData = new FormData();
    formData.append("file", file);
    loadingText.style.display = "block";

    try {
      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      localStorage.setItem("predictionData", JSON.stringify(data));
      window.location.href = "/result";
    } catch (err) {
      alert("Prediction failed: " + err.message);
    } finally {
      loadingText.style.display = "none";
    }
  });
});
