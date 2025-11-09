const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const FormData = require("form-data");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({ dest: uploadDir });

app.use(express.static(path.join(__dirname, "public")));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: form,
    });

    const text = await response.text(); // 👈 read raw response
    console.log("Flask raw response:", text); // 👈 log it

    let result;
    try {
      result = JSON.parse(text); // try to parse JSON
    } catch (err) {
      throw new Error("Flask did not return valid JSON:\n" + text);
    }

    fs.unlinkSync(req.file.path);
    res.json(result);
  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(5001, () => console.log("🚀 Node server running on port 5001"));
