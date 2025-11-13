/**
 * Script to drop the old unique index on the 'name' field
 * Run this if you're getting duplicate key errors
 */
require("dotenv").config();
const mongoose = require("mongoose");

const URL = process.env.URL;

async function clearIndex() {
  try {
    console.log("Connecting to MongoDB...");
    await mongoose.connect(URL);
    console.log("✅ Connected!");

    const db = mongoose.connection.db;
    const collection = db.collection('patients');

    // List all indexes
    const indexes = await collection.indexes();
    console.log("\nCurrent indexes:", JSON.stringify(indexes, null, 2));

    // Try to drop the unique index on 'name'
    try {
      await collection.dropIndex('name_1');
      console.log("\n✅ Successfully dropped 'name_1' index");
    } catch (err) {
      if (err.code === 27) {
        console.log("\n✅ Index 'name_1' doesn't exist (already removed or never created)");
      } else {
        console.error("\n❌ Error dropping index:", err.message);
      }
    }

    // Show indexes after dropping
    const indexesAfter = await collection.indexes();
    console.log("\nIndexes after cleanup:", JSON.stringify(indexesAfter, null, 2));

    await mongoose.connection.close();
    console.log("\n✅ Done! You can now save multiple records for the same patient.");
    process.exit(0);
  } catch (err) {
    console.error("❌ Error:", err.message);
    process.exit(1);
  }
}

clearIndex();
