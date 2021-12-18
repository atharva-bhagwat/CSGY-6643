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
- blocks: 19*11
- In project 2, the histogram intersection formula computes the similarity between the input image and the training image. 
The larger the similarity, the smaller the distance between the input image and the training image. 
"""

import os
import cv2
import numpy as np
from more_itertools import take

class HumanDetectorHOG():
    def __init__(self, image_data_path):
        self.image_data_path = image_data_path
        
        self.training_set = {}
        self.testing_set = {}

        self.cell_template_shape = (20,12)
        self.block_template_shape = (19,11)

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

        self.driver()

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

        # If angle is negative add 360 to make it positive
        gradient_angle[gradient_angle < 0] += 360

        # If angle is greater than 180, subtract 180
        gradient_angle[gradient_angle > 180] -= 180

        return np.pad(gradient_magnitude, 1), np.pad(gradient_angle, 1)

    def get_ratio(self, angle):
        if angle <= 10:
            lower_center = 10
            itr_lower = 0
            itr_upper = 8
        elif angle >=170:
            lower_center = 170
            itr_lower = 8
            itr_upper = 0
        else:
            for itr_i in range(9):
                if itr_i*20 + 10 < angle:
                    lower_center = itr_i*20 + 10
                    itr_lower = itr_i
                    itr_upper = itr_i + 1
        ratio = {itr_lower: 1 - abs(angle-lower_center)/20, itr_upper: abs(angle-lower_center)/20}
        return ratio

    def split_magnitude(self, magnitude, angle):
        ratio = self.get_ratio(angle)
        return {key: value*magnitude for key, value in ratio.items()}

    def get_hist(self, magnitude_slice, angle_slice):
        hist_bin = [0,0,0,0,0,0,0,0,0]
        for itr_i in range(magnitude_slice.shape[0]):
            for itr_j in range(magnitude_slice.shape[1]):
                magnitude_split = self.split_magnitude(magnitude_slice[itr_i][itr_j], angle_slice[itr_i][itr_j])
                for key, value in magnitude_split.items():
                    hist_bin[key] += value
        return hist_bin

    def calc_hist_bin(self, hist_bin_cellwise, gradient_magnitude, gradient_angle, cell_size = 8):
        for i in range(hist_bin_cellwise.shape[0]):
            for j in range(hist_bin_cellwise.shape[1]):
                magnitude_slice = gradient_magnitude[i*cell_size:(i*cell_size)+cell_size, j*cell_size:(j*cell_size)+cell_size]
                angle_slice = gradient_angle[i*cell_size:(i*cell_size)+cell_size, j*cell_size:(j*cell_size)+cell_size]
                hist_bin_cellwise[i][j] = self.get_hist(magnitude_slice, angle_slice)
        return hist_bin_cellwise

    def flatten(self, block):
        block_flat = np.array([])
        for itr_i in range(block.shape[0]):
            for itr_j in range(block.shape[1]):
                block_flat = np.append(block_flat, block[itr_i][itr_j])
        return block_flat

    def get_l2_norm(self, block_flat):
        return np.sqrt(sum(block_flat**2))

    def normalize_hist_bin(self, norm_hist_bin_blockwise, hist_bin_cellwise, block_size = 2):
        for itr_i in range(norm_hist_bin_blockwise.shape[0]):
            for itr_j in range(norm_hist_bin_blockwise.shape[1]):
                block = hist_bin_cellwise[itr_i:itr_i+block_size, itr_j:itr_j+block_size]
                block_flat = self.flatten(block)
                l2_norm = self.get_l2_norm(block_flat)
                norm_hist_bin_blockwise[itr_i][itr_j] = block_flat/l2_norm if l2_norm != 0 else block_flat
        return norm_hist_bin_blockwise

    def hog_driver(self, img):
        gradient_magnitude, gradient_angle = self.gradient_info_calc(img)
        hist_bin_cellwise = self.calc_hist_bin(np.empty(self.cell_template_shape, object), gradient_magnitude, gradient_angle)
        norm_hist_bin_blockwise = self.normalize_hist_bin(np.empty(self.block_template_shape, object), hist_bin_cellwise)
        descriptor = self.flatten(norm_hist_bin_blockwise)
        return descriptor

    def calc_similarity(self, test_descriptor, train_descriptor):
        sigma = 0
        for itr in range(len(test_descriptor)):
            sigma += min(train_descriptor[itr], test_descriptor[itr])
        return sigma/sum(train_descriptor)

    def predict(self, info):
        prediction = []
        for _, data in info.items():
            prediction.append(data['class'])
        return max(prediction, key=prediction.count)

    def knn(self, test_descriptor, k=3):
        neighbor_info = {}
        for image_filename, image_data in self.training_set.items():
            similarity = self.calc_similarity(test_descriptor, image_data['descriptor'])
            neighbor_info[image_filename] = {'similarity':similarity, 'class':image_data['class']}

        neighbor_info = {key: value for key, value in sorted(neighbor_info.items(), key=lambda value: value[1]['similarity'], reverse=True)}
        knn_info = dict(take(k, neighbor_info.items()))

        prediction = self.predict(knn_info)
        
        return prediction, knn_info

    def classify(self):
        for _, image_data in self.testing_set.items():
            image_data['predicted'], image_data['knn_info'] = self.knn(image_data['descriptor'])

    def load_data(self):
        for sub_folder in os.listdir(self.image_data_path):
            if os.path.isdir(os.path.join(self.image_data_path, sub_folder)):
                for image_filename in os.listdir(os.path.join(self.image_data_path, sub_folder)):
                        if '.bmp' in image_filename:
                            img = self.bgr_2_gray(self.read_img(os.path.join(self.image_data_path, sub_folder, image_filename)))
                            label = "Human" if 'pos' in sub_folder.lower() else "No-Human"
                            descriptor = self.hog_driver(img)
                            if 'training' in sub_folder.lower():
                                self.training_set[image_filename] = {'img':img,'class':label,'descriptor':descriptor}
                            if 'test' in sub_folder.lower():
                                self.testing_set[image_filename] = {'img':img,'actual':label,'descriptor':descriptor,'predicted':None,'knn_info':None}
    
    def driver(self):
        self.load_data()
        self.classify()

        for key, value in self.testing_set.items():
            print(f'{key}\nActual: {value["actual"]}\tPredicted: {value["predicted"]}\nInfo:\n{value["knn_info"]}\n\n**********\n\n')

if __name__ == '__main__':
    image_data_path = 'image_data'
    obj = HumanDetectorHOG(image_data_path)
