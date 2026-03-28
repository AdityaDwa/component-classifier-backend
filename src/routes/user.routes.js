import { Router } from "express";
import {
  changeCurrentPassword,
  deleteGuestUser,
  editProfileDetails,
  loginUser,
  logoutUser,
  refreshAccessToken,
  registerUser,
} from "../controllers/user.controller.js";
import { optionalAuth } from "../middlewares/auth.middleware.js";
const router = Router();
router.route("/register").post(registerUser);
router.route("/login").post(loginUser);
router.route("/change-password").patch(optionalAuth, changeCurrentPassword);
router.route("/edit-profile").patch(optionalAuth, editProfileDetails);
router.route("/logout").post(optionalAuth, logoutUser);
router.route("/refresh-token").patch(refreshAccessToken);
router.route("/guest").delete(deleteGuestUser);
export default router;
