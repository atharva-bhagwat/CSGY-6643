"""
Steps:
1) Read image. Convert to grayscale using: G = round(0.299R + 0.587G + 0.114B)
2) Gradient calculation (prewitt's operator), magnitude calculation (sqrt(Gx^2+Gy^2)). Normalize values. Compute gradient angle.
3) Calculate histogram bins for every cell (unsigned format).
4) Calculate normalized bins for every block.
5) Flatten and concat normalized bins for every block to get descriptor of length 7524.

Notes:
- Use 9 bins (unsigned). If gradient angle is greater than 180, subtract 180 from it. 
- Use copy of hist bin and add copies to ndarray to form cell output (20*11 ndarray). use this ndarray when parsing for blocks.
"""

import os
import cv2
import numpy as np

class HumanDetectorHOG():
    def __init__(self, image_data_path):
        self.image_data_path = image_data_path
        
        self.training_set = {}
        self.testing_set = {}

        self.cell_template_shape = (20,12)
        self.block_template_shape = (19,11)

        self.hist_bin = {
                            10:0,
                            30:0,
                            50:0,
                            70:0,
                            90:0,
                            110:0,
                            130:0,
                            150:0,
                            170:0,
                        }

        # Prewitt's horizontal gradient operator
        self.PREWITT_X = np.array(
                            [[-1,0,1],
                            [-1,0,1],
                            [-1,0,1]]
                        )
        # Prewitt's vertical gradient operator
        self.PREWITT_Y = np.array(
                            [[1,1,1],
                            [0,0,0],
                            [-1,-1,-1]]
                        )
        # Maximum possible value for gradient magnitude
        self.MAGNITUDE_NORMALIZATION_FACTOR = 3*(2**0.5)

    def read_img(self, path):
        return cv2.imread(path)
    
    def bgr_2_gray(self, img):
        blue, green, red = img[:,:,0], img[:,:,1], img[:,:,2]
        return np.around(0.299*red + 0.587*green + 0.114*blue)

    def convolution(self, x, y):
        """Implementation of convolution operation

        Args:
            x (ndarray): First numpy array
            y (ndarray): Second numpy array

        Returns:
            ndarray: Resultant of x*y; * -> convolution
        """
        x_shape = x.shape
        y_shape = y.shape
        output_shape = (x_shape[0]-y_shape[0]+1, x_shape[1]-y_shape[1]+1)
        output = np.zeros(output_shape)
        for itr_x in range(output_shape[0]):
            for itr_y in range(output_shape[1]):
                output[itr_x][itr_y] = (x[itr_x:itr_x+y_shape[0], itr_y:itr_y+y_shape[1]]*y).sum()
        return output

    def gradient_info_calc(self, img):
        gradient_x = self.convolution(img, self.PREWITT_X)
        gradient_y = self.convolution(img, self.PREWITT_Y)

        gradient_magnitude = np.hypot(gradient_x, gradient_y) # save this. need for report

        gradient_magnitude = gradient_magnitude/self.MAGNITUDE_NORMALIZATION_FACTOR

        gradient_angle = np.zeros(gradient_magnitude.shape)
        gradient_angle = np.rad2deg(np.arctan2(gradient_y, gradient_x))
        # If angle is negative add 180 to make it positive
        gradient_angle[gradient_angle < 0] += 180
        # If angle is greater than 180, subtract 180
        gradient_angle[gradient_angle > 180] -= 180

        return gradient_magnitude, gradient_angle

    def calc_hist_bin(self, hist_bin_cellwise, img, cell_size = 8):
        hist_bin = self.hist_bin.copy()
        for i in range(hist_bin_cellwise.shape[0]):
            for j in range(hist_bin_cellwise.shape[1]):
                pass        # write array slicing math in terms of i,j (cell locations) and pixels in img

    def normalize_hist_bin(self, norm_hist_bin_blockwise, hist_bin_cellwise, block_size = 2):
        pass

    def hog_driver(self, img):
        gradient_magnitude, gradient_angle = self.gradient_info_calc(img)
        hist_bin_cellwise = self.calc_hist_bin(np.zeros(self.cell_template_shape), img)
        norm_hist_bin_blockwise = self.normalize_hist_bin(np.zeros(self.block_template_shape), hist_bin_cellwise)

    def load_data(self):
        for sub_folder in os.listdir(self.image_data_path):
            if 'training' in sub_folder.lower():
                for image_filename in os.listdir(os.path.join(self.image_data_path, sub_folder)):
                    img = self.bgr_2_gray(self.read_img(os.path.join(self.image_data_path, sub_folder, image_filename)))
                    label = True if 'pos' in sub_folder.lower() else False
                    descriptor = self.hog_driver(img)
                    self.training_set[image_filename] = {'img':img,'class':label,'descriptor':descriptor}

            if 'test' in sub_folder.lower():
                for image_filename in os.listdir(os.path.join(self.image_data_path, sub_folder)):
                    img = self.bgr_2_gray(self.read_img(os.path.join(self.image_data_path, sub_folder, image_filename)))
                    label = True if 'pos' in sub_folder.lower() else False
                    descriptor = self.hog_driver(img)
                    self.training_set[image_filename] = {'img':img,'actual':label,'descriptor':descriptor}



if __name__ == '__main__':
    obj = HumanDetectorHOG()
