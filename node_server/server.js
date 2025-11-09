const express = require("express");
const multer = require("multer");
const fs = require("fs");
const FormData = require("form-data");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(express.static("public"));

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
