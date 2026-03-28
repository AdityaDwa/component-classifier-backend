import dotenv from "dotenv";
dotenv.config({ path: "./.env" });
const { default: connectDB } = await import("./db/index.js");
const { app } = await import("./app.js");
import { startCleanupJob } from "./utils/automaticGuestCleanup.js";
const port = process.env.PORT || 8000;
connectDB()
  .then(() => {
    app.on("error", (err) => {
      console.log("error:", err);
      throw err;
    });
    app.listen(port, () => {
      console.log(`Server is running at port ${port}`);
      startCleanupJob();
    });
  })
  .catch((error) => {
    console.log("MongoDB connection error", error);
  });
