const mongoose = require("mongoose");

// Schema for storing patient diagnosis results
const patientSchema = new mongoose.Schema({
  name: { type: String, required: true }, // Removed unique constraint
  prediction: { type: String, required: true },
  risk: { type: Number, required: true },
  change: { type: Number, required: true },
  confidence: { type: Number, default: 0 },
  scanDate: { type: Date, default: Date.now }, // Added scan timestamp
  lastTest: { type: String, required: true }, // Keep for backward compatibility
  chartData: { type: [Number], required: true },
}, {
  timestamps: true // Automatically adds createdAt and updatedAt
});

const Patient = mongoose.model("Patient", patientSchema);

module.exports = Patient;
