const mongoose = require("mongoose");

const URL = process.env.URL;

const connectDB = async () => {
  if (!URL) {
    console.error("❌ MongoDB connection failed: URL environment variable is not set!");
    console.error("💡 Please set the 'URL' environment variable in Railway with your MongoDB connection string");
    process.exit(1);
  }

  try {
    await mongoose.connect(URL);
    console.log("✅ MongoDB connected successfully!");
  } catch (err) {
    console.error("❌ MongoDB connection failed:", err.message);
    console.error("💡 Check your MongoDB connection string. Current URL:", URL.replace(/\/\/([^:]+):([^@]+)@/, '//$1:****@'));
    process.exit(1);
  }
};

module.exports = connectDB;
