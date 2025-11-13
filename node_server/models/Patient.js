const mongoose = require("mongoose");

// Schema for storing patient diagnosis results
const patientSchema = new mongoose.Schema({
  name: { type: String, required: true, unique: true },
  prediction: { type: String, required: true },
  risk: { type: Number, required: true },
  change: { type: Number, required: true },
  lastTest: { type: String, required: true },
  chartData: { type: [Number], required: true },
});

const Patient = mongoose.model("Patient", patientSchema);

module.exports = Patient;
