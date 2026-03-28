import express from "express";
import cookieParser from "cookie-parser";
import cors from "cors";

const app = express();

app.use(
  cors({
    origin: process.env.CORS_ORIGIN,
    credentials: true,
  })
);
//configurations
app.use(express.json({ limit: "16kb" }));
app.use(express.urlencoded({ limit: "16kb" }));
app.use(cookieParser());
app.use(express.static("public"));

//routes
import userRouter from "./routes/user.routes.js";
app.use("/api/v1/users", userRouter);
import imageRouter from "./routes/image.routes.js";
app.use("/api/v1/images", imageRouter);
import { errorHandler } from "./middlewares/errorHandler.middleware.js";
app.use(errorHandler);

export { app };
