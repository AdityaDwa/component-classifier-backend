import mongoose, { Schema } from "mongoose";
const componentSchema = new Schema(
  {
    id: {
      type: Number,
    },
    class: {
      type: String,
    },
    confidence: {
      type: Number,
    },
    bbox: {
      x: { type: Number },
      y: { type: Number },
      width: { type: Number },
      height: { type: Number },
    },
    analyzed_for: [{ type: String }],
    colors: {
      foreground_hex: { type: String },
      background_hex: { type: String },
      contrast_ratio: { type: Number },
      wcag_compliant: { type: Boolean },
      has_text: { type: Boolean },
      text_content: { type: String },
    },
    issues: [{ type: String }],
    filtered_out: { type: Boolean },
    filter_reason: { type: String },
  },
  {
    _id: false,
  }
);
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
    savedName: {
      type: String,
      default: "",
    },
    overall_grade: {
      type: String,
    },
    summary: {
      type: String,
    },
    metadata: {
      total_components_detected: { type: Number },
      components_in_layout_analysis: { type: Number },
      components_in_color_analysis: { type: Number },
      analysis_timestamp: { type: String },
    },
    components: [componentSchema],
    clutter:{
      score: { type: Number },
      category: { type: String },
      breakdown: {
        density: { type: Number },
        area_ratio: { type: Number },
        spacing_variance: { type: Number },
        overlap_penalty: { type: Number },
      },
      issues: [{ type: String }],
      suggestions: [{ type: String }],
    },
    alignment: {
      score: { type: Number },
      category: { type: String },
      breakdown: {
        left_edge: { type: Number },
        center: { type: Number },
        baseline: { type: Number },
      },
      issues: [{ type: String }],
      suggestions: [{ type: String }],
    },
    contrast: {
      average_contrast: { type: Number },
      compliant_count: { type: Number },
      total_text_components: { type: Number },
      compliance_rate: { type: Number },
      failed_components: [
        {
          component_id: { type: Number },
          class: { type: String },
          text_content: { type: String },
          contrast_ratio: { type: Number },
          threshold: { type: Number },
          foreground_hex: { type: String },
          background_hex: { type: String },
          foreground_rgb: [{ type: Number }],
          background_rgb: [{ type: Number }],
          issue: { type: String },
        },
      ],
      suggestions: [{ type: String }],
    },
  },
  { timestamps: true }
);
export const Image = mongoose.model("Image", imageSchema);
