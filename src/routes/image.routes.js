import { Router } from "express";
import { optionalAuth } from "../middlewares/auth.middleware.js";
import { upload } from "../middlewares/multer.middleware.js";
import {
  deleteEvaluation,
  getSavedEvaluations,
  getUIImage,
  saveEvaluation,
} from "../controllers/image.controller.js";

const router = Router();
router.route("/upload").post(optionalAuth, upload.single("files"), getUIImage);
router.route("/save/:imageId").patch(optionalAuth, saveEvaluation);
router.route("/saved").get(optionalAuth,getSavedEvaluations);
router.route("/:imageId").delete(optionalAuth, deleteEvaluation);

export default router;
