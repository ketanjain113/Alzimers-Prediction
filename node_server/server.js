require("dotenv").config();
const express = require("express");
const multer = require("multer");
const connectDB = require("./config/db");
const Patient = require("./models/Patient");
const fs = require("fs");
const path = require("path");
const FormData = require("form-data");
const fetch = (...args) => import("node-fetch").then(({ default: fetch }) => fetch(...args));


const app = express();

// ✅ Middleware
app.use(express.json());

// ✅ Connect to MongoDB
connectDB();

// ✅ Ensure uploads folder exists
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ✅ Multer config
const upload = multer({ dest: uploadDir });

// ✅ Serve Frontend (robust path resolution)
const candidateFrontendDirs = [
  path.join(__dirname, "..", "Frontend"),
  path.join(process.cwd(), "Frontend"),
  path.join(__dirname, "Frontend"),
];

let frontendDir = candidateFrontendDirs.find(
  (dir) => fs.existsSync(dir) && fs.statSync(dir).isDirectory()
);

if (frontendDir) {
  console.log(`📁 Serving frontend from: ${frontendDir}`);
  app.use(express.static(frontendDir));

  app.get("/", (req, res) => {
    res.sendFile(path.join(frontendDir, "index.html"));
  });
} else {
  console.warn("⚠️ Frontend directory not found. Expected one of:", candidateFrontendDirs);
  app.get("/", (req, res) => {
    res.status(404).send(
      `<h1>Frontend not available</h1><p>Make sure the Frontend folder is present next to the project root.</p>`
    );
  });
}

// ✅ Flask Model API URL
const MODEL_API_URL = process.env.MODEL_API_URL || "http://127.0.0.1:5000";

// ✅ Prediction Route (uploads image → forwards to Flask → saves to DB)
app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Send multipart form-data with field 'image'." });
    }

    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    // Forward to Flask API
    const response = await fetch(`${MODEL_API_URL}/predict`, {
      method: "POST",
      body: form,
    });

    const text = await response.text();
    console.log("Flask raw response:", text);

    let result;
    try {
      result = JSON.parse(text);
    } catch {
      throw new Error("Flask did not return valid JSON:\n" + text);
    }

    // Check if Flask returned an error
    if (result.error) {
      throw new Error(`Model API error: ${result.error}`);
    }

    // 🧠 Map Flask response to our format
    const confidence = result.confidence || 0;
    const risk = Math.round(confidence * 100); // Convert confidence to risk percentage
    const probabilities = result.probabilities || [];
    
    // Calculate change based on class probabilities (simplified)
    const change = probabilities.length >= 2 ? 
      Math.round((probabilities[0] - probabilities[1]) * 100) : 0;
    
    // Generate chart data from probabilities
    const chartData = probabilities.length > 0 ? 
      probabilities.map(p => Math.round(p * 100)) : 
      [40, 60, 75, 80, 90];

    // 🧠 Save prediction result to MongoDB
    const patientName = req.body.name || "unknown";
    const newPatient = new Patient({
      name: patientName.toLowerCase(),
      prediction: result.prediction || "Unknown",
      risk: risk,
      change: change,
      confidence: confidence,
      scanDate: new Date(),
      lastTest: new Date().toISOString().slice(0, 10),
      chartData: chartData,
    });

    try {
      await newPatient.save();
      console.log("✅ Saved new patient scan:", newPatient);
    } catch (saveErr) {
      console.error("⚠️ MongoDB save failed (continuing anyway):", saveErr.message);
      // Continue even if save fails - still return prediction to user
    }

    // Cleanup temp file
    fs.unlinkSync(req.file.path);

    // Return formatted response for frontend
    res.json({ 
      message: "Prediction completed and saved!", 
      prediction: result.prediction,
      confidence: confidence,
      risk: risk,
      change: change,
      chartData: chartData,
      patient: newPatient 
    });
  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ✅ Health Check Route (Flask)
app.get("/api/health", async (req, res) => {
  try {
    const resp = await fetch(`${MODEL_API_URL}/health`);
    const txt = await resp.text();

    if (!resp.ok) {
      return res.status(502).json({ error: `Model health check failed: ${resp.status}`, body: txt });
    }

    try {
      const json = JSON.parse(txt);
      return res.json({ upstream: json });
    } catch {
      return res.json({ upstream_text: txt });
    }
  } catch (err) {
    console.error("Health proxy error:", err.message);
    res.status(502).json({ error: `Unable to reach model at ${MODEL_API_URL}: ${err.message}` });
  }
});

// ✅ Fetch ALL records for a patient by name (for insights/history)
app.get("/api/patient/:name/history", async (req, res) => {
  try {
    const name = req.params.name.trim();
    // Find all records for this patient, sorted by date (newest first)
    const records = await Patient.find({ name: new RegExp(`^${name}$`, "i") })
      .sort({ scanDate: -1 });
    
    if (!records || records.length === 0) {
      return res.status(404).json({ message: "No records found for this patient" });
    }

    // Get the latest record for current stats
    const latestRecord = records[0];

    // Prepare timeline data for graphing
    const timeline = records.reverse().map(record => ({
      date: new Date(record.scanDate).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      risk: record.risk,
      prediction: record.prediction,
      confidence: record.confidence,
      fullDate: record.scanDate
    }));

    res.json({
      name: latestRecord.name,
      totalScans: records.length,
      latestScan: {
        prediction: latestRecord.prediction,
        risk: latestRecord.risk,
        change: latestRecord.change,
        confidence: latestRecord.confidence,
        date: latestRecord.scanDate,
        lastTest: latestRecord.lastTest
      },
      timeline: timeline,
      allRecords: records
    });
  } catch (err) {
    console.error("Error fetching patient history:", err);
    res.status(500).json({ message: "Server error" });
  }
});

// ✅ Fetch a Patient by Name (single/latest record)
app.get("/api/patient/:name", async (req, res) => {
  try {
    const name = req.params.name.trim();
    // Find the most recent record for this patient
    const patient = await Patient.findOne({ name: new RegExp(`^${name}$`, "i") })
      .sort({ scanDate: -1 });
    
    if (!patient) return res.status(404).json({ message: "Patient not found" });
    res.json(patient);
  } catch (err) {
    console.error("Error fetching patient:", err);
    res.status(500).json({ message: "Server error" });
  }
});


// ✅ Add New Patient Manually (if needed)
app.post("/api/patient", async (req, res) => {
  try {
    const { name, prediction, risk, change, lastTest, chartData } = req.body;

    if (!name || !prediction || !risk || !change || !lastTest || !chartData) {
      return res.status(400).json({ error: "All fields are required" });
    }

    const newPatient = new Patient({
      name: name.toLowerCase(),
      prediction,
      risk,
      change,
      lastTest,
      chartData,
    });

    await newPatient.save();
    res.status(201).json({ message: "✅ Patient data added successfully", newPatient });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ✅ Run Node on Configurable Port
const PORT = process.env.PORT;
app.listen(PORT, () => {
  console.log(`🚀 Node server running on http://0.0.0.0:${PORT} (MODEL_API_URL=${MODEL_API_URL})`);
});
