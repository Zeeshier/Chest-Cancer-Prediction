const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const uploadMessage = document.getElementById('uploadMessage');
const predictButton = document.getElementById('predictButton');
const resultDiv = document.getElementById('result');

imageInput.addEventListener('change', function () {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.innerHTML = `<img src="${e.target.result}" alt="CT Scan Preview" />`;
      uploadMessage.textContent = "Image Loaded Successfully!";
      predictButton.disabled = false;  // Enable Predict button
    };
    reader.readAsDataURL(file);
  }
});

document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();
  
  const formData = new FormData();
  formData.append("file", imageInput.files[0]);
  
  resultDiv.textContent = "Predicting...";

  try {
    const res = await fetch("/api/chest_cancer/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (data.prediction) {
      resultDiv.textContent = `Result: ${data.prediction}`;
    } else {
      resultDiv.textContent = "Error: Could not predict.";
    }
  } catch (error) {
    console.error(error);
    resultDiv.textContent = "Server error. Please try again.";
  }
});
