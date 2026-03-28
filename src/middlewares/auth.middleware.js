import jwt from "jsonwebtoken";
import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { User } from "../models/user.model.js";
const optionalAuth = asyncHandler(async (req, res, next) => {
  const token = req.header("Authorization")?.replace("Bearer ", "");

  if (!token) {
    return next();
  }
  let decodedToken;
  try {
    decodedToken = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET);
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      throw new ApiError(401, "Access token expired.Please login again");
    }
    throw new ApiError(401, "Invalid access token");
  }
  const user = await User.findById(decodedToken._id).select(
    "-password -refreshToken"
  );
  if (!user) {
    throw new ApiError(401, "Invalid access token");
  }
  req.user = user;
  next();
});
export { optionalAuth };
