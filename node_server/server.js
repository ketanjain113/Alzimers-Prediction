const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const FormData = require("form-data");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();

// ✅ ensure uploads folder exists
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ✅ multer config
const upload = multer({ dest: uploadDir });

// ✅ serve frontend files
app.use(express.static(path.join(__dirname, "../Frontend")));


app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../Frontend', 'index.html'));
});


// ✅ image upload + forward to Python
const MODEL_API_URL = process.env.MODEL_API_URL || "http://127.0.0.1:5000";

app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    // forward to Flask API
    const response = await fetch(`${MODEL_API_URL}/predict`, {
      method: "POST",
      body: form,
    });

    const text = await response.text(); // raw Flask response
    console.log("Flask raw response:", text);

    let result;
    try {
      result = JSON.parse(text); // try parsing JSON
    } catch (err) {
      throw new Error("Flask did not return valid JSON:\n" + text);
    }

    // cleanup
    fs.unlinkSync(req.file.path);

    res.json(result);
  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ✅ Run Node on configurable port (Railway provides $PORT)
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`🚀 Node server running on http://0.0.0.0:${PORT} (MODEL_API_URL=${MODEL_API_URL})`));
