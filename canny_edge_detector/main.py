import os
import cv2
import numpy as np
from math import atan, degrees

"""
STEPS:
1) Read image in grayscale
2) Gaussian smoothing
3) Gradient calculation, magnitude calculation, gradient angle calculation
4) Non-maxima supperession
5) Thresholding: use simple thresholding and produce three binary edge maps by 
using three thresholds chosen at the 25th, 50th and 75th percentiles of the 
gradient magnitudes after non-maxima suppression

Notes: Input image and output image should be of same size. Replace undefined values with 0.

To save: 
1) Normalized  image result after Gaussian smoothing.
2) Normalized horizontal and vertical gradient responses (two separate images) 
    **To generate normalized gradient  responses, take the absolute value of the results first and then normalize**
3) Normalized gradient magnitude image.
4) Normalized gradient magnitude image after non-maximasuppression.
5) Binary edge maps using simple thresholding for thresholds chosen at the 25th, 50th and 75th percentiles.
"""

class CannyEdgeDetector():
    def __init__(self, img_file):
        """Initialise variables, constants

        Args:
            img_file (str): Image file name (.bmp)
        """
        self.img_file = img_file
        self.img_path = os.path.join('input_images',img_file)
        self.output_folder = 'out_folder'
        self.img = None
        self.smooth_img = None
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_angle = None
        self.gradient_magnitude = None
        self.magnitude_nms = None
        
        # Gaussian filter as per the question
        self.GAUSSIAN_FILTER = np.array(
                                    [[1,1,2,2,2,1,1],
                                    [1,2,2,4,2,2,1],
                                    [2,2,4,8,4,2,2],
                                    [2,4,8,16,8,4,2],
                                    [2,2,4,8,4,2,2],
                                    [1,2,2,4,2,2,1],
                                    [1,1,2,2,2,1,1]]
                                )

        # Sum of entires in the gaussian filter
        self.NORMALIZATION_FACTOR = 140

        # Prewitt's horizontal gradient operator
        self.PREWITT_X = np.array(
                            [[-1,0,1],
                            [-1,0,1],
                            [-1,0,1]]
                        )
        # Prewitt's vertical gradient ooperator
        self.PREWITT_Y = np.array(
                            [[1,1,1],
                            [0,0,0],
                            [-1,-1,-1]]
                        )
        # Call to driver function. Performs canny edge detection on input image.
        self.driver()
        
    def is_dir(self, directory):
        """Helper function to create directories if they dont exist

        Args:
            directory (str): Directory path
        """
        if not os.path.isdir(directory):
            os.mkdir(directory)
            print(f'Creating {directory}...')
            
    def read_img(self):
        """Read image stored at img_path in grayscale
        """
        self.img = cv2.imread(self.img_path, 0)
    
    def write_img(self, filename, file):
        """Saves image

        Args:
            filename (str): Filename to store images
            file (ndarray): Image numpy array
        """
        cv2.imwrite(os.path.join(self.output_folder,filename), file)
        print(f'{filename} saved...')
        
    def slice_array(self, array, start_x, start_y, filter_length):
        """Returns array slices to convolute

        Args:
            array (ndarray): Input array
            start_x (int): Left top corner x coordinate to start slicing
            start_y (int): Left top corner y coordinate to start slicing
            filter_length (int): Length of the filter to stop slicing

        Returns:
            ndarray: Sliced array
        """
        slice = []
        for itr_i in range(start_x, start_x+filter_length):
            row = []
            for itr_j in range(start_y, start_y+filter_length):
                row.append(array[itr_i][itr_j])
            slice.append(row)
        return np.array(slice)
    
    def convolution(self, x, y):
        """Implementation of convolution operation

        Args:
            x (ndarray): First numpy array
            y (ndarray): Second numpy array

        Returns:
            ndarray: Resultant of x*y; * -> convolution
        """
        output = np.zeros(x.shape)
        start_x = start_y = 1
        for i in range(x.shape[0]-len(y)+1):
            for j in range(x.shape[1]-len(y)+1):
                output[start_x,start_y] = (self.slice_array(x, start_x-1,start_y-1, y.shape[0])*y).sum()
                start_y += 1
            start_x += 1
            start_y = 1
        return output
    
    def angle_calc(self):
        """Calculates gradient angle: tan inv (Gy/Gx)
        """
        self.gradient_angle = np.zeros(self.gradient_magnitude.shape)
        for itr_x in range(self.gradient_angle.shape[0]):
            for itr_y in range(self.gradient_angle.shape[1]):
                if self.gradient_x[itr_x][itr_y] != 0:
                    self.gradient_angle[itr_x][itr_y] = degrees(atan(self.gradient_y[itr_x][itr_y]/self.gradient_x[itr_x][itr_y]))
                else:
                    self.gradient_angle[itr_x][itr_y] = 0
        
    def gaussian_smoothing(self):
        """Gaussian smoothing operation: IMAGE * GAUSSIAN_FILTER; * -> convolution
        """
        self.smooth_img = self.convolution(self.img, self.GAUSSIAN_FILTER)/self.NORMALIZATION_FACTOR
        self.write_img('smooth_'+self.img_file, self.smooth_img)
        
    def gradient_calc(self):
        """Calculate horizontal and vertical gradients using prewitt operator: IMAGE * PREWITT_X and IMAGE * PREWITT_y; * -> convolution
        """
        self.gradient_x = self.convolution(self.smooth_img, self.PREWITT_X)
        self.gradient_y = self.convolution(self.smooth_img, self.PREWITT_Y)
        self.gradient_magnitude = np.sqrt(np.square(self.gradient_x)+np.square(self.gradient_y))
        self.write_img('horizontal_'+self.img_file, self.gradient_x)
        self.write_img('vertical_'+self.img_file, self.gradient_y)
        self.write_img('magnitude_'+self.img_file, self.gradient_magnitude)
        self.angle_calc()
        
    def nms(self):
        pass
    
    def driver(self):
        """Calls canny edge detection functions in order:
            - Read image
            - Gaussian smoothing
            - Gradient calculation using prewitt's operator
            - Non-maxima supperession
            - Thresholding
        """
        self.is_dir(self.output_folder)
        self.read_img()
        self.gaussian_smoothing()
        self.gradient_calc()
        # TODO:
        # - Non-maxima supperession
        # - Thresholding
        self.nms()
        
        
if __name__ == '__main__':
    obj = CannyEdgeDetector('coins.bmp')