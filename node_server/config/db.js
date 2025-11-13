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

    // Drop old unique index on 'name' field if it exists
    try {
      const Patient = mongoose.connection.collection('patients');
      await Patient.dropIndex('name_1');
      console.log("✅ Dropped old unique index on 'name' field");
    } catch (indexErr) {
      // Index doesn't exist or already dropped, that's fine
      if (indexErr.code !== 27) { // 27 = IndexNotFound
        console.log("ℹ️ No old index to drop (this is normal)");
      }
    }
  } catch (err) {
    console.error("❌ MongoDB connection failed:", err.message);
    console.error("💡 Check your MongoDB connection string. Current URL:", URL.replace(/\/\/([^:]+):([^@]+)@/, '//$1:****@'));
    process.exit(1);
  }
};

module.exports = connectDB;
