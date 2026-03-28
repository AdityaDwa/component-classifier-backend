import { v2 as cloudinary } from "cloudinary";
import fs from "fs";
//configurations

const uploadOnCloudinary = async (localFilePath) => {
  try {
    if (!localFilePath) {
      return null;
    }
    cloudinary.config({
      cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
      api_key: process.env.CLOUDINARY_API_KEY,
      api_secret: process.env.CLOUDINARY_API_SECRET,
    });
    const response = await cloudinary.uploader.upload(localFilePath, {
      resource_type: "auto",
    });
    console.log("File has been uploaded on cloudinary", response.url);
    fs.unlinkSync(localFilePath);
    return response;
  } catch (error) {
    console.log(error.message);
    fs.unlinkSync(localFilePath); //remove the locally saved temp file as uploading failed
    return null;
  }
};
const deleteFromCloudinary = async (publicId) => {
  try {
    if (!publicId) {
      return null;
    }
    cloudinary.config({
      cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
      api_key: process.env.CLOUDINARY_API_KEY,
      api_secret: process.env.CLOUDINARY_API_SECRET,
    });
    const response = await cloudinary.uploader.destroy(publicId);
    console.log("File deleted from cloudinary successfully.");
    return response;
  } catch (error) {
    console.log("Error deleting image from cloudinary", error.message);
    return null;
  }
};
export { uploadOnCloudinary, deleteFromCloudinary };
