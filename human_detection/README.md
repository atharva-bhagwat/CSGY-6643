# Human Detection using HOG Feature

## Steps:
1) Read image. Convert to grayscale using: Gray = round(0.299*Red + 0.587*Green + 0.114*Blue)
2) Gradient calculation using prewitt's operator, magnitude calculation (sqrt(Gx^2+Gy^2)) and normalization. Gradient angle calculation.
3) Calculate histogram bins for every cell (unsigned format/9 bins).
4) Calculate normalized bins for every block (l2 norm).
5) Flatten and concatenate normalized bins for every block to get a descriptor of length 7524.
6) 3NN implementations using similarity formula: sum(min(input, train))/sum(train)

## Usage:
`python3 main.py <path_to_image_folder>`