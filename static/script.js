document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const imageInput = document.getElementById("imageInput");
  if (!imageInput.files.length) return;

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);

  const resultDiv = document.getElementById("result");
  resultDiv.textContent = "Analyzing... Please wait.";

  try {
    const res = await fetch("http://localhost:5000/predict", {
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
    resultDiv.textContent = "Server error. Try again.";
  }
});
