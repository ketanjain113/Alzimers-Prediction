// seed.js
const mongoose = require("mongoose");
const Patient = require("./models/Patient"); // adjust path if needed
const connectDB = require("./config/db"); // adjust path if needed

// Connect to DB
connectDB();

const seedData = [
  { name: "Aarav Sharma", prediction: "Healthy", risk: 10, change: 2, lastTest: "2025-11-01", chartData: [10, 20, 25, 15, 18] },
  { name: "Priya Mehta", prediction: "Mild Cognitive Impairment", risk: 30, change: 5, lastTest: "2025-10-28", chartData: [30, 35, 40, 38, 42] },
  { name: "Rohan Gupta", prediction: "Alzheimer's Disease (Early)", risk: 45, change: 7, lastTest: "2025-09-22", chartData: [45, 48, 50, 55, 60] },
  { name: "Ishita Verma", prediction: "Healthy", risk: 12, change: 1, lastTest: "2025-10-10", chartData: [12, 13, 11, 10, 14] },
  { name: "Aditya Singh", prediction: "Alzheimer's Disease (Moderate)", risk: 65, change: 8, lastTest: "2025-11-05", chartData: [60, 63, 65, 68, 70] },
  { name: "Neha Patel", prediction: "Healthy", risk: 8, change: 0, lastTest: "2025-08-15", chartData: [8, 9, 7, 10, 8] },
  { name: "Karan Jain", prediction: "Mild Cognitive Impairment", risk: 35, change: 4, lastTest: "2025-10-02", chartData: [30, 33, 35, 36, 37] },
  { name: "Ananya Rao", prediction: "Alzheimer's Disease (Advanced)", risk: 80, change: 10, lastTest: "2025-09-30", chartData: [75, 77, 78, 80, 82] },
  { name: "Manav Kapoor", prediction: "Healthy", risk: 15, change: 2, lastTest: "2025-10-19", chartData: [10, 12, 14, 15, 16] },
  { name: "Simran Kaur", prediction: "Mild Cognitive Impairment", risk: 40, change: 3, lastTest: "2025-10-25", chartData: [38, 39, 40, 42, 41] }
];

async function seedDatabase() {
  try {
    await Patient.deleteMany({});
    console.log("🗑️ Cleared old data");

    await Patient.insertMany(seedData);
    console.log("✅ Inserted new sample patients");

    mongoose.connection.close();
    console.log("🔒 Database connection closed");
  } catch (err) {
    console.error("❌ Error seeding data:", err);
  }
}

seedDatabase();
