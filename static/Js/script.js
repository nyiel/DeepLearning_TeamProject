document.addEventListener("DOMContentLoaded", () => {
    const openCameraBtn = document.getElementById("openCameraBtn");
    const video = document.getElementById("cameraStream");
    const canvas = document.getElementById("cameraCanvas");
    const captureBtn = document.getElementById("captureBtn");
    const preview = document.getElementById("preview");
    const fileInput = document.getElementById("imageUpload");
    const uploadForm = document.getElementById("upload-form");
    const loadingDiv = document.getElementById("loading");
    let stream = null;

    // 1. Open Camera
    openCameraBtn.addEventListener("click", async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: "environment" } // Prefer back camera on mobile
            });
            video.srcObject = stream;
            video.style.display = "block";
            captureBtn.style.display = "inline-block";
            openCameraBtn.style.display = "none";
            preview.style.display = "none";
        } catch (err) {
            alert("Could not access camera: " + err);
        }
    });

    // 2. Capture Photo
    captureBtn.addEventListener("click", () => {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to Blob -> File
        canvas.toBlob((blob) => {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            
            // Create a DataTransfer to update the file input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            // Show Preview
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
            
            // Stop Stream
            stopCamera();
        }, "image/jpeg");
    });

    // 3. Stop Camera Helper
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.style.display = "none";
            captureBtn.style.display = "none";
            openCameraBtn.style.display = "inline-flex";
        }
    }

    // 4. Handle Gallery Selection Preview
    fileInput.addEventListener("change", function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = "block";
                stopCamera(); // Close camera if user switches to gallery
            }
            reader.readAsDataURL(this.files[0]);
        }
    });

    // 5. Show Loading on Submit
    uploadForm.addEventListener("submit", () => {
        loadingDiv.style.display = "block";
    });
});