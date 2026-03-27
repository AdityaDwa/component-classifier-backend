"""
Main analysis orchestrator - combines all layout and color metrics.

This is the entry point for running complete UI quality analysis on a webpage.
Coordinates preprocessing, layout analysis, and color contrast checks, then
assembles a comprehensive report with both per-component detail and aggregate
page-level metrics.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from analysis.preprocessing import ComponentPreprocessor
from analysis.layout_analysis import LayoutAnalyzer
from analysis.color_analysis import ColorAnalyzer


class UIAnalyzer:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.path_utils = PathUtils()

        self.preprocessor = ComponentPreprocessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.color_analyzer = ColorAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_layout(
        self,
        image_path: str,
        yolo_detections: List[Dict],
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ) -> Dict:
        """
        Complete UI quality analysis pipeline.

        Pipeline:
          1. Assign sequential component_id to every raw detection.
          2. Filter structural containers (page wrappers >80 % of viewport).
          3. Calculate clutter score — density, area, spacing, overlap.
          4. Calculate alignment score — left edge, center, baseline variance.
          5. Identify text-bearing components (heading, text, button, link,
             input, navigation).
          6. Run OCR + K-means color extraction on text-bearing components.
          7. Assess WCAG AA contrast compliance per text component.
          8. Enrich components_map with per-component issues and color data.
          9. Calculate overall grade (A–D) and summary sentence.
         10. Save timestamped JSON to analysis_results/.

        Args:
            image_path (str): Absolute path to the webpage screenshot.
            yolo_detections (List[Dict]): Raw YOLO predictions. Each dict must
                contain at minimum:
                    {
                        "class": "button",
                        "confidence": 0.92,
                        "bbox": {"x": 100, "y": 200, "width": 150, "height": 80}
                    }
                Note: a "component_id" key is assigned by this method — callers
                should NOT set it before calling analyze_layout().
            viewport_width (int): Screenshot width in pixels. Default 1920.
            viewport_height (int): Screenshot height in pixels. Default 1080.

        Returns:
            Dict with the following top-level keys:

            "metadata":
                image_path, viewport dimensions, total_components_detected,
                components_in_layout_analysis, components_in_color_analysis,
                analysis_timestamp.

            "components" — list sorted by id, one entry per detected component:
                {
                    "id": 3,
                    "class": "button",
                    "confidence": 0.92,
                    "bbox": {"x": 100, "y": 200, "width": 150, "height": 80},

                    "filtered_out": false,
                    "filter_reason": null,
                    -- filtered_out=true means the component is a structural
                       page-wrapper container excluded from all analysis.
                       filter_reason explains why.

                    "analyzed_for": ["layout", "color"],
                    -- "layout"  present if the component passed the structural
                                filter and was included in clutter + alignment.
                    -- "color"   present if the component is a text-bearing class
                                (heading, text, button, link, input, navigation)
                                and was passed to OCR + contrast analysis.

                    "colors": {
                        "foreground_hex": "#333333",
                        "background_hex": "#FFFFFF",
                        "contrast_ratio": 12.6,
                        "wcag_compliant": true,
                        "has_text": true,
                        "text_content": "Sign Up"
                    },
                    -- colors is null when:
                        (a) the component is not text-bearing,
                        (b) OCR found no text inside the component,
                        (c) image crop / color extraction failed.

                    "issues": [
                        "overlaps with component #5 (input, IoU=0.14)",
                        "WCAG AA contrast fail: 3.20:1 (fg #999999 on bg #EEEEEE,
                                                         minimum 4.5:1 required)"
                    ]
                    -- issues is an empty list if no problems detected.
                }

            "clutter"   — aggregate score, category, breakdown, issues, suggestions.
            "alignment" — aggregate score, category, breakdown, issues, suggestions.
            "contrast"  — aggregate compliance stats, failed_components list,
                          suggestions. (per_component_data has been removed here;
                          individual color detail lives in "components".)
            "overall_grade" — "A" | "B" | "C" | "D"
            "summary"       — one-sentence human-readable description.
        """
        self.logger.info("inside analyze_layout method..........")
        self.logger.info(
            f"Analyzing {len(yolo_detections)} detections from {image_path}"
        )

        try:
            # ------------------------------------------------------------------
            # Step 1: Assign sequential component IDs
            #
            # Every detection receives an integer ID before anything else runs.
            # Because all downstream functions receive the same dict objects
            # (Python passes dicts by reference), the IDs are visible everywhere
            # without any additional passing — layout_analysis.py reads them when
            # building overlap_pairs, color_analysis.py reads them when building
            # per_component_data.
            # ------------------------------------------------------------------
            for idx, det in enumerate(yolo_detections):
                det["component_id"] = idx

            # Build the working map.  Every component starts with sensible
            # defaults; we enrich each entry progressively as analysis runs.
            components_map: Dict[int, Dict] = {
                det["component_id"]: {
                    "id": det["component_id"],
                    "class": det.get("class", "unknown"),
                    "confidence": round(float(det.get("confidence", 0.0)), 4),
                    "bbox": det.get("bbox", {}),
                    "filtered_out": False,
                    "filter_reason": None,
                    "analyzed_for": [],
                    "colors": None,
                    "issues": [],
                }
                for det in yolo_detections
            }

            # ------------------------------------------------------------------
            # Step 2: Filter structural containers
            #
            # Containers that cover >80 % of the viewport are page-level wrappers
            # (e.g. <div class="page-wrapper">). Including them in clutter and
            # alignment calculations would artificially inflate scores because
            # they enclose every other component.
            #
            # Filtered components are flagged in components_map so the frontend
            # knows they exist but were excluded from analysis.
            # ------------------------------------------------------------------
            filtered_detections = self.preprocessor.filter_structural_containers(
                yolo_detections, viewport_width, viewport_height
            )

            filtered_ids = {det["component_id"] for det in filtered_detections}

            for cid, comp in components_map.items():
                if cid not in filtered_ids:
                    comp["filtered_out"] = True
                    comp["filter_reason"] = (
                        "structural container — covers >80 % of viewport "
                        "(page wrapper; excluded from layout and color analysis)"
                    )

            self.logger.info(
                f"After filtering: {len(filtered_detections)} components "
                f"({len(yolo_detections) - len(filtered_detections)} removed)"
            )

            # All components that survived filtering are analyzed for layout.
            for det in filtered_detections:
                components_map[det["component_id"]]["analyzed_for"].append("layout")

            # ------------------------------------------------------------------
            # Step 3: Clutter Score
            #
            # calculate_clutter_score() returns an "overlap_pairs" key containing
            # a list of dicts, each with:
            #   "component1" / "component2" — full detection dicts (carry component_id)
            #   "iou"                       — Intersection over Union value
            #   …class names, bboxes (used by _group_overlaps_by_region internally)
            #
            # We pop overlap_pairs here and use it to attach per-component overlap
            # issues.  After the pop, "overlap_pairs" is absent from clutter_result
            # and will NOT appear in the client-facing JSON.
            # ------------------------------------------------------------------
            clutter_result = self.layout_analyzer.calculate_clutter_score(
                filtered_detections, viewport_width, viewport_height
            )

            for pair in clutter_result.pop("overlap_pairs", []):
                det_i = pair["component1"]
                det_j = pair["component2"]
                iou_val = pair["iou"]
                cid_i = det_i.get("component_id")
                cid_j = det_j.get("component_id")

                if cid_i is not None and cid_i in components_map:
                    components_map[cid_i]["issues"].append(
                        f"overlaps with component #{cid_j} "
                        f"({det_j.get('class', 'unknown')}, IoU={iou_val:.2f})"
                    )
                if cid_j is not None and cid_j in components_map:
                    components_map[cid_j]["issues"].append(
                        f"overlaps with component #{cid_i} "
                        f"({det_i.get('class', 'unknown')}, IoU={iou_val:.2f})"
                    )

            # ------------------------------------------------------------------
            # Step 4: Alignment Score
            #
            # Alignment is a geometric aggregate — there is no per-component data
            # to extract here. Issues are reported at the layout level (in the
            # "alignment" key of the final report, not in individual components).
            # ------------------------------------------------------------------
            alignment_result = self.layout_analyzer.calculate_alignment_score(
                filtered_detections, viewport_width, viewport_height
            )

            # ------------------------------------------------------------------
            # Step 5 & 6 & 7: Color Contrast Analysis
            #
            # Only text-bearing component classes undergo OCR and color extraction:
            #   heading, text, button, link, input, navigation
            #
            # assess_contrast_compliance() returns a "per_component_data" key
            # alongside the existing aggregate keys. per_component_data contains
            # one entry per processed detection with:
            #   component_id, foreground_hex, background_hex, contrast_ratio,
            #   wcag_compliant, has_text, text_content
            # (foreground_hex / background_hex are None when no text was found.)
            #
            # We pop per_component_data here, attach it to components_map, and
            # also surface WCAG failures as per-component issues. After the pop,
            # "per_component_data" is absent from contrast_result and will NOT
            # appear in the client-facing JSON.
            # ------------------------------------------------------------------
            text_bearing_components = self.preprocessor.filter_text_bearing_components(
                filtered_detections
            )

            for det in text_bearing_components:
                components_map[det["component_id"]]["analyzed_for"].append("color")

            contrast_result = self.color_analyzer.assess_contrast_compliance(
                image_path, text_bearing_components
            )

            for comp_color in contrast_result.pop("per_component_data", []):
                cid = comp_color.get("component_id")
                if cid is None or cid not in components_map:
                    continue

                if comp_color["has_text"] and comp_color["foreground_hex"] is not None:
                    # Populate color detail for this component.
                    components_map[cid]["colors"] = {
                        "foreground_hex": comp_color["foreground_hex"],
                        "background_hex": comp_color["background_hex"],
                        "contrast_ratio": comp_color["contrast_ratio"],
                        "wcag_compliant": comp_color["wcag_compliant"],
                        "has_text": True,
                        "text_content": comp_color["text_content"],
                    }
                    # Surface WCAG failure as a component-level issue.
                    if not comp_color["wcag_compliant"]:
                        components_map[cid]["issues"].append(
                            f"WCAG AA contrast fail: "
                            f"{comp_color['contrast_ratio']:.2f}:1 "
                            f"(foreground {comp_color['foreground_hex']} on "
                            f"background {comp_color['background_hex']}, "
                            f"minimum 4.5:1 required)"
                        )
                # else: colors stays None (no text found or extraction failed)

            # ------------------------------------------------------------------
            # Step 8: Overall Grade and Summary
            # ------------------------------------------------------------------
            overall_grade, summary = self._calculate_overall_grade(
                clutter_result, alignment_result, contrast_result
            )

            # ------------------------------------------------------------------
            # Step 9: Assemble Final Report
            #
            # components list is sorted by id for deterministic output.
            # ------------------------------------------------------------------
            components_list = [
                components_map[cid] for cid in sorted(components_map.keys())
            ]

            report = {
                "metadata": {
                    "image_path": image_path,
                    "viewport_width": viewport_width,
                    "viewport_height": viewport_height,
                    "total_components_detected": len(yolo_detections),
                    "components_in_layout_analysis": len(filtered_detections),
                    "components_in_color_analysis": len(text_bearing_components),
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "components": components_list,
                "clutter": clutter_result,
                "alignment": alignment_result,
                "contrast": contrast_result,
                "overall_grade": overall_grade,
                "summary": summary,
            }

            self.logger.info(
                f"Analysis complete — overall grade: {overall_grade}"
            )
            self._save_report(report)
            return report

        except Exception as e:
            self.logger.exception(f"Error in analyze_layout: {e}")
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_overall_grade(
        self, clutter: Dict, alignment: Dict, contrast: Dict
    ) -> tuple:
        """
        Calculates overall letter grade (A–D) and summary text from three scores.

        Grading criteria:
            Clutter  : lower is better (0–30=A, 30–50=B, 50–70=C, 70+=D)
            Alignment: higher is better (>0.85=A, >0.65=B, >0.45=C, else D)
            Contrast : WCAG compliance rate (>0.9=A, >0.7=B, >0.5=C, else D)

        Args:
            clutter (Dict): Result from calculate_clutter_score().
            alignment (Dict): Result from calculate_alignment_score().
            contrast (Dict): Result from assess_contrast_compliance().

        Returns:
            Tuple[str, str]: (letter_grade, summary_sentence)
        """
        clutter_score = clutter.get("score", 50)
        alignment_score = alignment.get("score", 0.5)
        contrast_compliance = contrast.get("compliance_rate", 0.0)

        grades = []

        if clutter_score < 30:
            grades.append("A")
        elif clutter_score < 50:
            grades.append("B")
        elif clutter_score < 70:
            grades.append("C")
        else:
            grades.append("D")

        if alignment_score > 0.85:
            grades.append("A")
        elif alignment_score > 0.65:
            grades.append("B")
        elif alignment_score > 0.45:
            grades.append("C")
        else:
            grades.append("D")

        if contrast_compliance > 0.9:
            grades.append("A")
        elif contrast_compliance > 0.7:
            grades.append("B")
        elif contrast_compliance > 0.5:
            grades.append("C")
        else:
            grades.append("D")

        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1}
        avg = sum(grade_values[g] for g in grades) / len(grades)

        if avg >= 3.5:
            overall_grade = "A"
        elif avg >= 2.5:
            overall_grade = "B"
        elif avg >= 1.5:
            overall_grade = "C"
        else:
            overall_grade = "D"

        clutter_desc = clutter.get("category", "moderate")
        alignment_desc = alignment.get("category", "acceptable")
        total_text = contrast.get("total_text_components", 0)
        compliant = contrast.get("compliant_count", 0)
        compliance_pct = int(contrast_compliance * 100)

        summary = (
            f"Overall grade: {overall_grade}. "
            f"Layout is {clutter_desc}ly cluttered with {alignment_desc} alignment. "
            f"Color contrast: {compliance_pct}% WCAG AA compliant "
            f"({compliant}/{total_text} components)."
        )

        return overall_grade, summary

    def _save_report(self, report: Dict) -> None:
        """
        Saves the analysis report to a timestamped JSON file in analysis_results/.

        Args:
            report (Dict): Complete analysis report dict.

        Returns:
            None
        """
        try:
            results_dir = self.path_utils.get_base_path().joinpath(
                "analysis_results"
            )
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"layout_report_{timestamp}.json"
            filepath = results_dir.joinpath(filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Report saved to {filepath}")

        except Exception as e:
            self.logger.exception(f"Error saving report: {e}")