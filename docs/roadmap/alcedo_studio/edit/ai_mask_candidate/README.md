# AI Mask candidate models (#458)

Reports of the model re-evaluation for the AI Segmentation Mask. The steps and the evidence rules
are in [the master plan, section 8.16.4](../ai_segmentation_mask_master_plan.md#8164-exploration-steps-458).
Weights, ONNX files, and full-size outputs stay under `build/tmp/explore/` and are not committed.

The comparison images of the reports are under `images/`. They show photos that must not be
shared again, so `images/` is in `.gitignore` and is not distributed with the GitHub repository.
The image links in the reports work only on a machine that has the images; each report states the
command that makes them from the local evidence.

| Step | Report | Status |
| --- | --- | --- |
| E1 | [SAM 2.1 point and box prompts](e1_sam2_1_point_box_evaluation.md) | Measured 2026-10-19; user review open |
| E2 | [BiRefNet one-click Subject and matting](e2_birefnet_subject_matting.md) | Measured 2026-10-19; stopped by the user; only lite kept |
| E3 | Sky and People body-part models | Not started |
| E4 | [Click selection with candidates and hover highlight](e4_click_candidates.md) | Measured 2026-10-19; user review open |
| E5 | Edge stage | Only if E2 is not sufficient |
| E6 | Report per tool and recommendation | Not started |
