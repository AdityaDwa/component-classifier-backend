import mongoose from "mongoose";
import { ApiError } from "./ApiError.js";
const parseObjectId = (idArray = []) => {
  return idArray.map((id, index) => {
    if (!mongoose.Types.ObjectId.isValid(id)) {
      throw new ApiError(400, `Invalid ObjectId: ${id}`);
    }
    return new mongoose.Types.ObjectId(id); //though use of new deprecates ObjectId, without new we get error
    //Class constructor Objectid cannot be invoked without "new"
  });
};
export { parseObjectId };
