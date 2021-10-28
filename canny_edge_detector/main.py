import os
import argparse
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
4) Normalized gradient magnitude image after non-maxima suppression.
5) Binary edge maps using simple thresholding for thresholds chosen at the 25th, 50th and 75th percentiles.
"""

class CannyEdgeDetector():
    def __init__(self, img_filename):
        """Initialise variables, constants

        Args:
            img_filename (str): Image filename (.bmp)
        """
        self.img_filename = img_filename
        self.img_path = os.path.join('input_images',img_filename)
        self.output_folder = os.path.join('out_folder', self.img_filename.split('.')[0])
        self.img = None
        self.smooth_img = None
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_angle = None
        self.gradient_magnitude = None
        self.quantized_angle = None
        self.magnitude_nms = None
        self.edgemap_t25 = None
        self.edgemap_t50 = None
        self.edgemap_t75 = None
        
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

        # Sector map to quantize gradient angles depending on multiple
        self.SECTORS = {
            0:0,
            1:1,
            2:1,
            3:2,
            4:2,
            5:3,
            6:3,
            7:0,
            8:0
        }

        # Map to calculate neighbors depending on the sector
        self.NEIGHBORS = {
            0:{'l':(-1,0),'r':(1,0)},
            1:{'l':(-1,1),'r':(1,-1)},
            2:{'l':(0,-1),'r':(0,1)},
            3:{'l':(-1,-1),'r':(1,1)}
        }

        # Call to driver function. Performs canny edge detection on input image.
        self.driver()
        
    def is_dir(self, directory):
        """Helper function to create directories if they dont exist

        Args:
            directory (str): Directory path
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
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
        out_path = os.path.join(self.output_folder,filename)
        cv2.imwrite(out_path, file)
        print(f'{out_path} saved...')
        
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
        for _ in range(x.shape[0]-len(y)+1):
            for _ in range(x.shape[1]-len(y)+1):
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
                    self.gradient_angle[itr_x][itr_y] = np.NAN
    
    def get_sector(self, angle):
        """Returns sector value (0-3)
            Logic: As the sector wheel is same along the 0-180 degree line, 
            we reduce the range from 0-360 to 0-180 for the gradient angle
            We achieve this by subtracting it by 180 if it is greater than 180
            In case where the gradient angle is -ve
            we first add 360 to it and then subtract it by 180 if it is greater than 180
            We then divide this value by 22.5(as the sector is divided into smaller sectors of 22.5 degrees)
            This value gives us the multiple which will futher give us the sector using map
            Usage of map reduces the complexity of the problem

        Args:
            angle (float): Gradient angle for a pixel location

        Returns:
            int: Sector value (0-3)
        """
        if angle < 0:
            angle += 360
        if angle > 180:
            angle -= 180
        angle = int(angle//22.5)
        return self.SECTORS[angle]

    def quantize_angle(self):
        """Function iterates over the gradient angle array and calculates sector
            for every pixel location. If the 3*3 mask goes outside the edges of the image,
            the quantized value is set to nan
        """
        self.quantized_angle = np.zeros(self.gradient_magnitude.shape)
        for itr_x in range(self.quantized_angle.shape[0]):
            for itr_y in range(self.quantized_angle.shape[1]):
                if not np.isnan(self.gradient_angle[itr_x][itr_y]):
                    self.quantized_angle[itr_x][itr_y] = self.get_sector(self.gradient_angle[itr_x][itr_y])
                else:
                    self.quantized_angle[itr_x][itr_y] = np.NAN

    def nms_compare(self, ind_x, ind_y, sector):
        """Fuction calculates neighbor coordinates and checks if the center pixel is maximum out of the neighbors
            If the center pixel is the maximum, then function returns the magnitude of the pixel
            Else returns 0

        Args:
            ind_x (int): X coordinate of the center pixel
            ind_y (int): Y coordinate of the center pixel
            sector (int): Quantized gradient angle

        Returns:
            int: Gradient magnitude or 0
        """
        neighbor_l = {'x':ind_x+self.NEIGHBORS[sector]['l'][0],'y':ind_y+self.NEIGHBORS[sector]['l'][1]}
        neighbor_r = {'x':ind_x+self.NEIGHBORS[sector]['r'][0],'y':ind_y+self.NEIGHBORS[sector]['r'][1]}
        if self.gradient_magnitude[ind_x][ind_y] == max(self.gradient_magnitude[ind_x][ind_y], self.gradient_magnitude[neighbor_l['x']][neighbor_l['y']], self.gradient_magnitude[neighbor_r['x']][neighbor_r['y']]):
            return self.gradient_magnitude[ind_x][ind_y]
        else:
            return 0

    def apply_threshold(self, x, threshold):
        """Applies simple thresholding operation
            If magnitude < threshold, set the edgemap pixel value to 0
            Else set the edgemap pixel value to 255 (white pixel)

        Args:
            x (ndarray): Gradient magnitude after non-maxima suppression
            threshold (float): Threshold value

        Returns:
            ndarray: Edgemap created after applying simple threshold operation
        """
        edge_map = np.zeros(x.shape)
        for itr_x in range(x.shape[0]):
            for itr_y in range(x.shape[1]):
                if x[itr_x][itr_y] < threshold:
                    edge_map[itr_x][itr_y] = 0
                else:
                    edge_map[itr_x][itr_y] = 255
        return edge_map

    def gaussian_smoothing(self):
        """Gaussian smoothing operation: IMAGE * GAUSSIAN_FILTER; * -> convolution
        """
        self.smooth_img = self.convolution(self.img, self.GAUSSIAN_FILTER)/self.NORMALIZATION_FACTOR
        self.write_img('smooth_'+self.img_filename, self.smooth_img)
        
    def gradient_calc(self):
        """Calculate horizontal and vertical gradients using prewitt operator: IMAGE * PREWITT_X and IMAGE * PREWITT_y; * -> convolution
        """
        self.gradient_x = self.convolution(self.smooth_img, self.PREWITT_X)
        self.gradient_y = self.convolution(self.smooth_img, self.PREWITT_Y)
        self.gradient_magnitude = np.sqrt(np.square(self.gradient_x)+np.square(self.gradient_y))
        self.write_img('horizontal_'+self.img_filename, self.gradient_x)
        self.write_img('vertical_'+self.img_filename, self.gradient_y)
        self.write_img('magnitude_'+self.img_filename, self.gradient_magnitude)
        self.angle_calc()
        
    def nms(self):
        """Non-maxima suppression function
            Steps:
                1) Quantize gradient angles
                2) Iterate over gradient magnitude and compare center pixel with neighbors
        """
        self.quantize_angle()
        self.magnitude_nms = np.zeros(self.gradient_magnitude.shape)
        for itr_x in range(self.magnitude_nms.shape[0]):
            for itr_y in range(self.magnitude_nms.shape[1]):
                if not np.isnan(self.quantized_angle[itr_x][itr_y]):
                    self.magnitude_nms[itr_x][itr_y] = self.nms_compare(itr_x, itr_y, self.quantized_angle[itr_x][itr_y])
                else:
                    self.magnitude_nms[itr_x][itr_y] = 0
        self.write_img('nms_magnitude_'+self.img_filename, self.magnitude_nms)

    def thresholding(self):
        """Simple thresholding operation
            We calculate 3 thresholds:
                1) T_25 = 25th percentile of magnitude after nms
                2) T_50 = 50th percentile of magnitude after nms
                3) T_75 = 75th percentile of magnitude after nms
        """
        threshold_25 = np.quantile(list(set(self.magnitude_nms.flatten())),0.25)
        threshold_50 = np.quantile(list(set(self.magnitude_nms.flatten())),0.50)
        threshold_75 = np.quantile(list(set(self.magnitude_nms.flatten())),0.75)

        self.edgemap_t25 = self.apply_threshold(self.magnitude_nms, threshold_25)
        self.edgemap_t50 = self.apply_threshold(self.magnitude_nms, threshold_50)
        self.edgemap_t75 = self.apply_threshold(self.magnitude_nms, threshold_75)

        self.write_img('edgemap_t25_'+self.img_filename, self.edgemap_t25)
        self.write_img('edgemap_t50_'+self.img_filename, self.edgemap_t50)
        self.write_img('edgemap_t75_'+self.img_filename, self.edgemap_t75)
    
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
        self.nms()
        self.thresholding()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Canny Edge Detector.')
    parser.add_argument('img_filename', type=str, help='image filename')
    args = parser.parse_args()
    obj = CannyEdgeDetector(args.img_filename)