# Canny Edge Detector

## Steps:
1) Gaussian Smoothing using a 7*7 mask
2) Gradient calculation using Prewitt's Operator
3) Non-maxima Suppression
4) Simple Thresholding [25th, 50th, 75th Percentile]

## Usage:
`python3 main.py <path_to_input_file>`
