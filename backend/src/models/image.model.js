import mongoose, { Schema } from "mongoose";
const imageSchema = new Schema(
  {
    imageUrl: {
      type: String,
      required: true,
    },
    cloudinaryPublicId: {
      type: String,
    },
    uploadedBy: {
      type: Schema.Types.ObjectId,
      ref: "User",
    },
    isSaved: {
      type: Boolean,
      default: false,
    },
    savedName:{
      type:String,
      default:""
    },
    scores: {
      type: Object,
    },

    components: {
      type: [
        {
          id: Number,
          label: String,
          x: Number,
          y: Number,
          height: Number,
          width: Number,
        },
      ],
    },
  },
  { timestamps: true }
);
export const Image = mongoose.model("Image", imageSchema);
