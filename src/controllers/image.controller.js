import { asyncHandler } from "../utils/asyncHandler.js";
import { Image } from "../models/image.model.js";
import { uploadOnCloudinary } from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { parseObjectId } from "../utils/parseObjectId.js";
import { deleteFromCloudinary } from "../utils/cloudinary.js";
import { User } from "../models/user.model.js";
import axios from "axios";

const getUIImage = asyncHandler(async (req, res) => {
  const imageLocalPath = req.file?.path;
  const guestId = req.headers["x-guest-id"];
  console.log(imageLocalPath);
  if (!imageLocalPath) {
    throw new ApiError(403, "UI image is compulsory to upload");
  }
  let userId;
  if (req.user) {
    userId = req.user._id;
  } else {
    if (!guestId) {
      throw new ApiError(403, "Guest id is necessary");
    }

    let guestUser = await User.findOne({ guestId, isGuest: true });
    if (!guestUser) {
      guestUser = await User.create({
        guestId,
        isGuest: true,
      });
      if (!guestUser) {
        throw new ApiError(500, "Guest user instance not created");
      }
    }
    userId = guestUser._id;
  }
  const image = await uploadOnCloudinary(imageLocalPath);
  if (!image) {
    throw new ApiError(403, "UI image is compulsory to upload");
  }
  const mlResponse = await axios.post(
    `${process.env.ML_SERVICE_URL}/evaluate`,
    { imageUrl: image.url }
  );
  const { scores, components } = mlResponse.data;
  const storedImage = await Image.create({
    imageUrl: image.url,
    cloudinaryPublicId: image.public_id,
    uploadedBy: userId,
    scores,
    components,
  });
  if (!storedImage) {
    throw new ApiError(500, "UI image storage unsuccessful.");
  }
  return res
    .status(200)
    .json(new ApiResponse(200, storedImage, "UI image stored successfully."));
});
const saveEvaluation = asyncHandler(async (req, res) => {
  const { imageId } = req.params;
  const { savedName } = req.body;
  const [realImageObjectId] = parseObjectId([imageId]);
  const image = await Image.findById(realImageObjectId);
  if (!image) {
    throw new ApiError(404, "Image not found");
  }
  const updatedImage = await Image.findByIdAndUpdate(
    realImageObjectId,
    {
      isSaved: true,
      savedName,
    },
    { new: true }
  );
  if (!updatedImage) {
    throw new ApiError(500, "Error while saving the evaluation");
  }
  return res
    .status(200)
    .json(new ApiResponse(200, updatedImage, "Evaluation saved successfully"));
});
const getSavedEvaluations = asyncHandler(async (req, res) => {
  const savedEvaluations = await Image.find({
    isSaved: true,
    uploadedBy: req.user._id,
  });
  return res
    .status(200)
    .json(
      new ApiResponse(
        200,
        savedEvaluations,
        "Saved Evaluations fetched successfully."
      )
    );
});
const deleteEvaluation = asyncHandler(async (req, res) => {
  const { imageId } = req.params;
  const [RealImageObjectId] = parseObjectId([imageId]);
  const image = await Image.findById(RealImageObjectId);
  await deleteFromCloudinary(image.cloudinaryPublicId);
  const deletedImage = await Image.findByIdAndDelete(RealImageObjectId);
  if (!deletedImage) {
    throw new ApiError(500, "Error while deleting image from database.");
  }
  return res
    .status(200)
    .json(new ApiResponse(200, deletedImage, "Image deleted successfully."));
});
export { getUIImage, saveEvaluation,getSavedEvaluations ,deleteEvaluation};
