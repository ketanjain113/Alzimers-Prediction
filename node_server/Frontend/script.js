// =============================
// 🧠 ALZHEIMER PREDICTION LOGIC
// =============================

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const fileInput = document.getElementById("image");
  const nameInput = document.getElementById("name");
  const resultBox = document.getElementById("result");

  // ✅ If page doesn’t have the form (e.g., home/about), skip setup
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = fileInput?.files[0];
    const name = nameInput?.value.trim();

    if (!file || !name) {
      alert("⚠️ Please enter your name and upload an MRI image!");
      return;
    }

    // Prepare multipart form data
    const formData = new FormData();
    formData.append("image", file);
    formData.append("name", name);

    try {
      resultBox.innerHTML = "⏳ Analyzing MRI scan... please wait.";

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const text = await response.text(); // handle non-JSON edge cases
      let data;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Invalid JSON from server: " + text);
      }

      if (!response.ok) {
        throw new Error(data.error || "Server returned an error");
      }

      // ✅ Display prediction result
      resultBox.innerHTML = `
        <h3>🧠 Prediction Result</h3>
        <p><b>Patient:</b> ${name}</p>
        <p><b>Prediction:</b> ${data.prediction}</p>
        <p><b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%</p>
        <p><b>Risk Level:</b> ${data.risk || 0}%</p>
        <p><b>Status:</b> ${data.change >= 0 ? "📈 Getting Worse" : "📉 Improving"}</p>
      `;

    } catch (err) {
      console.error("❌ Prediction Error:", err);
      resultBox.innerHTML = `
        <p style="color:red;">❌ ${err.message || "Failed to connect to server."}</p>
      `;
    }
  });
});
