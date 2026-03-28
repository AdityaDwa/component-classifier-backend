import cron from "node-cron";
import { User } from "../models/user.model.js";
import { Image } from "../models/image.model.js";
import { deleteFromCloudinary } from "../utils/cloudinary.js";

export const startCleanupJob = () => {
  cron.schedule("*/30 * * * *", async () => {
    console.log("Running guest cleanup job...");
    const expiredGuestUsers = await User.find({
      isGuest: true,
      createdAt: { $lt: new Date(Date.now() - 30 * 60 * 1000) },
    });

    for (const guestUser of expiredGuestUsers) {
      const images = await Image.find({ uploadedBy: guestUser._id });
      for (const image of images) {
        if (image.cloudinaryPublicId) {
          await deleteFromCloudinary(image.cloudinaryPublicId);
        }
      }
      await Image.deleteMany({ uploadedBy: guestUser._id });
      await User.findByIdAndDelete(guestUser._id);
    }
  });
};
