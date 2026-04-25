# Single Thread Channel Hierarchy

Threshold-first workflow for extracting multiscale breakpoints from a river centerline using:
- sinuosity
- turning energy
- mean deviation from the original line

The current branch keeps threshold detection and representative mode visualization separate:
- thresholds are selected from peaks in the boundary score
- visualization modes can be chosen either from stable intervals or directly from threshold sigmas

## Future Work

- Replace the current single `width_m` near-original flag with a variable-width formulation along the reach. This would make the "too similar to original" diagnostic more defensible for rivers that widen or narrow substantially downstream.
- Revisit peak detection so candidate generation is less permissive at very small scales.
- Add reach-heterogeneity checks before applying a single threshold hierarchy to a long mixed-style centerline.
