import { User } from "../models/user.model.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { Image } from "../models/image.model.js";
import { deleteFromCloudinary } from "../utils/cloudinary.js";
const generateAccessAndRefreshTokens = async (userId) => {
  try {
    const user = await User.findById(userId);
    const accessToken = user.generateAccessToken();
    const refreshToken = user.generateRefreshToken();
    user.refreshToken = refreshToken;
    await user.save({ validateBeforeSave: false });
    return { accessToken, refreshToken };
  } catch (error) {
    throw new ApiError(
      500,
      "Error while generating access and refresh tokens."
    );
  }
};
const registerUser = asyncHandler(async (req, res) => {
  const { username, email, password } = req.body;
  const guestId = req.headers["x-guest-id"];
  if ([username, email, password].some((elem) => elem.trim() === "")) {
    throw new ApiError(403, "All fields are compulsory to fill.");
  }
  const existingUser = await User.findOne({
    $or: [{ username }, { email }],
  });
  if (existingUser) {
    let message = "Duplicate entry. ";
    if (existingUser.username === username) {
      message += "Username already exists.";
    }
    if (existingUser.email === email) {
      message += "Email already exists.";
    }
    throw new ApiError(409, message);
  }
  const user = await User.create({
    username,
    email,
    password,
    isActive: true,
    isGuest: false,
  });
  if (!user) {
    throw new ApiError(500, "User cannot be registered successfully.");
  }
  if (guestId) {
    const guestUser = await User.findOne({ guestId, isGuest: true });

    if (guestUser) {
      // move all images linked to guest user to newly registered user
      await Image.updateMany(
        { uploadedBy: guestUser._id },
        {
          $set: {
            uploadedBy: user._id,
          },
        }
      );
      // delete guest user
      await User.findByIdAndDelete(guestUser._id);
    }
  }

  const createdUser = await User.findById(user._id).select(
    "-password -refreshToken"
  );
  res
    .status(201)
    .json(new ApiResponse(201, createdUser, "User registered successfully"));
});
const loginUser = asyncHandler(async (req, res) => {
  const { email, password } = req.body;
  const guestId = req.headers["x-guest-id"];
  if (!email) {
    throw new ApiError(403, "Email is required");
  }
  const user = await User.findOne({ email });
  if (!user) {
    throw new ApiError(404, "User not found");
  }
  const isPasswordCorrect = await user.isPasswordCorrect(password);
  if (!isPasswordCorrect) {
    throw new ApiError(401, "Invalid user credentials");
  }
  const { accessToken, refreshToken } = await generateAccessAndRefreshTokens(
    user._id
  );
  if (guestId) {
    const guestUser = await User.findOne({ guestId, isGuest: true });

    if (guestUser) {
      // move all images linked to guest user to logged in user
      await Image.updateMany(
        { uploadedBy: guestUser._id },
        {
          $set: {
            uploadedBy: user._id,
          },
        }
      );
      // delete guest user
      await User.findByIdAndDelete(guestUser._id);
    }
  }
  const options = {
    httpOnly: true,
    secure: true,
  };
  return res
    .status(200)
    .cookie("accessToken", accessToken, options)
    .cookie("refreshToken", refreshToken, options)
    .json(
      new ApiResponse(
        200,
        { accessToken, refreshToken },
        "User logged in successfully."
      )
    );
});
const logoutUser = asyncHandler(async (req, res) => {
  const user = await User.findByIdAndUpdate(
    req.user._id,
    {
      $set: { refreshToken: undefined },
    },
    { new: true }
  ).select("-password -refreshToken");
  const options = {
    httpOnly: true,
    secure: true,
  };
  return res
    .status(200)
    .clearCookie("accessToken", options)
    .clearCookie("refreshToken", options)
    .json(new ApiResponse(200, user, "User logged out successfully"));
});
const changeCurrentPassword = asyncHandler(async (req, res) => {
  const { current_password, new_password, confirmed_newpassword } = req.body;
  const user = await User.findById(req.user._id);
  const isCurrentPasswordValid = await user.isPasswordCorrect(current_password);
  if (!isCurrentPasswordValid) {
    throw new ApiError(401, "Current password is incorrect.");
  }
  if (new_password !== confirmed_newpassword) {
    throw new ApiError(403, "New password and confirmed password dont match.");
  }
  if (current_password === new_password) {
    throw new ApiError(
      400,
      "Current password and new password fields are the same.Nothing to update"
    );
  }
  user.password = new_password;
  await user.save({ validateBeforeSave: false });
  return res
    .status(200)
    .json(new ApiResponse(200, {}, "Password changed successfully."));
});
const editProfileDetails = asyncHandler(async (req, res) => {
  console.log(req.body);
  const { email, username } = req.body;
  const query = {};
  if (!(username || email)) {
    throw new ApiError(
      403,
      "Provide at least one of the editable profile parameters"
    );
  }
  if (email) {
    const existing = await User.findOne({ email, _id: { $ne: req.user._id } });
    if (existing) {
      throw new ApiError(409, "Email already in use");
    }
    query.email = email;
  }
  if (username) {
    const existing = await User.findOne({
      username,
      _id: { $ne: req.user._id },
    });
    if (existing) {
      throw new ApiError(409, "Username already in use");
    }
    query.username = username;
  }
  const updatedUser = await User.findByIdAndUpdate(req.user._id, query, {
    new: true,
  }).select("-password -refreshToken");
  if (!updatedUser) {
    throw new ApiError(500, "Editing profile details unsuccessful");
  }
  res
    .status(200)
    .json(
      new ApiResponse(200, updatedUser, "Editing profile details successful")
    );
});
const refreshAccessToken = asyncHandler(async (req, res) => {
  const incomingRefreshToken = req.body.refreshToken;
  if (!incomingRefreshToken) {
    throw new ApiError(401, "Invalid refresh token");
  }
  let decodedRefreshToken;
  try {
    decodedRefreshToken = jwt.verify(
      incomingRefreshToken,
      process.env.REFRESH_TOKEN_SECRET
    );
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      throw new ApiError(
        401,
        "Refresh token expired. Please login in to continue."
      );
    }
    throw new ApiError(401, "Invalid refresh token");
  }
  const user = await User.findById(decodedRefreshToken?._id);
  if (!user) {
    throw new ApiError(401, "Invalid refresh token");
  }
  if (incomingRefreshToken !== user?.refreshToken) {
    throw new ApiError(401, "Refresh token is expired or used");
  }
  const { accessToken, refreshToken: newRefreshToken } =
    await generateAccessAndRefreshTokens(user._id);
  const options = {
    httpOnly: true,
    secure: true,
  };
  return res
    .status(200)
    .cookie("accessToken", accessToken, options)
    .cookie("refreshToken", newRefreshToken, options)
    .json(
      new ApiResponse(
        200,
        { accessToken, refreshToken: newRefreshToken },
        "Access token refreshed"
      )
    );
});
const deleteGuestUser = asyncHandler(async (req, res) => {
  const guestId = req.headers["x-guest-id"];
  if (!guestId) {
    throw new ApiError(403, "Guest account is necessary.");
  }
  const guestUser = await User.findOne({ guestId, isGuest: true });
  if (!guestUser) {
    return res
      .status(200)
      .json(new ApiResponse(200, {}, "Nothing to clean up"));
  }
  const images = await Image.find({ uploadedBy: guestUser._id });
  for (const image of images) {
    if (image.cloudinaryPublicId) {
      await deleteFromCloudinary(image.cloudinaryPublicId);
    }
  }
  await Image.deleteMany({ uploadedBy: guestUser._id });
  await User.findByIdAndDelete(guestUser._id);
  return res
    .status(200)
    .json(new ApiResponse(200, {}, "Guest data cleaned up successfully"));
});
export {
  registerUser,
  loginUser,
  logoutUser,
  changeCurrentPassword,
  editProfileDetails,
  refreshAccessToken,
  deleteGuestUser,
};
