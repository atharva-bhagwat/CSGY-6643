import cv2
import numpy as np
from math import atan2, degrees

gau = np.array([[1,1,2,2,2,1,1],
                [1,2,2,4,2,2,1],
                [2,2,4,8,4,2,2],
                [2,4,8,16,8,4,2],
                [2,2,4,8,4,2,2],
                [1,2,2,4,2,2,1],
                [1,1,2,2,2,1,1]]
            )
PREWITT_X = np.array(
                    [[-1,0,1,0,0],
                    [-1,0,1,0,0],
                    [-1,0,1,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]]
                )
        # Prewitt's vertical gradient operator
PREWITT_Y = np.array(
                    [[1,1,1,0,0],
                    [0,0,0,0,0],
                    [-1,-1,-1,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]]
                )



def slice_array(array, start_x, start_y, filter_length):
    slice = []
    for itr_i in range(start_x, start_x+filter_length):
        row = []
        for itr_j in range(start_y, start_y+filter_length):
            row.append(array[itr_i][itr_j])
        slice.append(row)
    return np.array(slice)

boundary = 0

def convolution(x, y):
    global boundary
    output = np.zeros(x.shape)
    start_x = start_y = (len(y)//2) + boundary
    for _ in range(x.shape[0]-len(y)+1):
        for _ in range(x.shape[1]-len(y)+1):
            output[start_x][start_y] = (slice_array(x, start_x-(len(y)//2)-boundary,start_y-(len(y)//2)-boundary, y.shape[0])*y).sum()
            # print(f'slice:{slice_array(x, start_x-(len(y)//2),start_y-(len(y)//2), y.shape[0])}')
            # print(f'filter: {y}')
            # print(f'res: {output[start_x][start_y]}\nstartx: {start_x}, starty: {start_y}')
            start_y += 1
        start_x += 1
        start_y = len(y)//2
    boundary += len(y)//2
    return output

def convolution2(x, y):
    output = np.zeros(x.shape)
    start_x = start_y = (len(y)//2)
    for itr_x in range(x.shape[0]-len(y)+1):
        for itr_y in range(x.shape[1]-len(y)+1):
            output[start_x][start_y] = (x[itr_x:itr_x+y.shape[0], itr_y:itr_y+y.shape[1]]*y).sum()
            # print(f'slice:{slice_array(x, start_x-(len(y)//2),start_y-(len(y)//2), y.shape[0])}')
            # print(f'filter: {y}')
            # print(f'res: {output[start_x][start_y]}\nstartx: {start_x}, starty: {start_y}')
            start_y += 1
        start_x += 1
        start_y = len(y)//2
    return output

def conv(x,y):
    ir, ic = x.shape
    fr, fc = y.shape
    rr, rc = ir - fr + 1, ic - fc + 1
    z = np.zeros((rr,rc))
    for i in range(rr):
        for j in range(rc):
            # print(f'conv_slice:{x[i:i+fr,j:j+fc]}')
            # print(f'conv_filter: {y}')
            z[i][j] = np.sum(x[i:i+fr,j:j+fc]*y)
            # print(f'conv_res: {z[i][j]}\ni: {i}, j: {j}')
    return z

img = cv2.imread('input_images/House.bmp',0)
# img = cv2.imread('input_images/test_patterns.bmp',0)
pad = 0
si = conv(img,gau)/140
pad += len(gau)//2
si_pad = np.pad(si,pad)
si2 = convolution2(img, gau)/140

print(f'si: {si.shape}\nsi2: {si2.shape}')

# for i in range(si.shape[0]):
#     for j in range(si.shape[1]):
#         if si[i][j] != si2[i][j]:
#             print(f'i:{i}, j:{j}, si:{si[i][j]}, si2:{si2[i][j]}')

# print(np.array_equal(si,si2))

cv2.imwrite('si.bmp',si)
cv2.imwrite('si_pad.bmp',si_pad)
cv2.imwrite('si2.bmp',si2)

ix = conv(si,PREWITT_X)
pad += len(PREWITT_X)//2
ix_pad = np.pad(ix,pad)

ix2 = convolution2(si2,PREWITT_X)

print(f'ix: {ix.shape}\nix2: {ix2.shape}')

# for i in range(ix.shape[0]):
#     for j in range(ix.shape[1]):
#         if ix[i][j] != ix2[i][j]:
#             print(f'i:{i}, j:{j}, ix:{ix[i][j]}, ix2:{ix2[i][j]}')

# print(np.array_equal(ix,ix2))

cv2.imwrite('ix.bmp',ix)
cv2.imwrite('ix_pad.bmp',ix_pad)
cv2.imwrite('ix2.bmp',ix2)

iy = conv(si,PREWITT_Y)
iy_pad = np.pad(iy,pad)

iy2 = convolution2(si2,PREWITT_Y)

print(f'iy: {iy.shape}\niy2: {iy2.shape}')

# for i in range(iy.shape[0]):
#     for j in range(iy.shape[1]):
#         if iy[i][j] != iy2[i][j]:
#             print(f'i:{i}, j:{j}, iy:{iy[i][j]}, iy2:{iy2[i][j]}')

# print(np.array_equal(iy,iy2))

cv2.imwrite('iy.bmp',iy)
cv2.imwrite('iy_pad.bmp',iy_pad)
cv2.imwrite('iy2.bmp',iy2)

g = np.sqrt(np.square(ix)+np.square(iy))
g_pad = np.sqrt(np.square(ix_pad)+np.square(iy_pad))
g2 = np.sqrt(np.square(ix2)+np.square(iy2))

print(f'g: {g.shape}\ng2: {g2.shape}')

print(f'si eq: {np.array_equal(si, si_pad)}')
print(f'ix eq: {np.array_equal(ix, ix_pad)}')
print(f'iy eq: {np.array_equal(iy, iy_pad)}')
print(f'g eq: {np.array_equal(g, g_pad)}')

# for i in range(g.shape[0]):
#     for j in range(g.shape[1]):
#         if g[i][j] != g2[i][j]:
#             print(f'i:{i}, j:{j}, g:{g[i][j]}, g2:{iy2[i][j]}')

# print(np.array_equal(g,g2))

cv2.imwrite('g.bmp',g)
cv2.imwrite('g_pad.bmp',g_pad)
cv2.imwrite('g2.bmp',g2)
gradient_angle = quantized_angle = quantized_angle_np = None
def angle_calc(gradient_magnitude,gradient_x,gradient_y):
    global gradient_angle
    gradient_angle = np.zeros(gradient_magnitude.shape)
    gradient_angle = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    # for itr_x in range(gradient_angle.shape[0]):
    #     for itr_y in range(gradient_angle.shape[1]):
    #         gradient_angle[itr_x][itr_y] = degrees(atan2(gradient_y[itr_x][itr_y],gradient_x[itr_x][itr_y]))
            
    # print(angtest)
    # print(gradient_angle)
    # print(f'angeq: {np.array_equal(gradient_angle, angtest)}')
                                                   
SECTORS = {
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
NEIGHBORS = {
            0:{'l':(-1,0),'r':(1,0)},
            1:{'l':(-1,1),'r':(1,-1)},
            2:{'l':(0,-1),'r':(0,1)},
            3:{'l':(-1,-1),'r':(1,1)}
        }

def get_sector(angle):
    """Returns sector value (0-3)
        Logic: As the sector wheel is same along the 0-180 degree line, 
        we reduce the range from 0-360 to 0-180 for the gradient angle
        We achieve this by subtracting it by 180 if it is greater than 180
        In case where the gradient angle is -ve
        we first add 360 to it and then subtract it by 180 if it is greater than 180
        We then divide this value by 22.5(as the sector is divided into smaller sectors of 22.5 degrees)
        This value gives us the multiple which will further give us the sector using map
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
    return SECTORS[angle]

def quantize_angle(gradient_magnitude):
    """Function iterates over the gradient angle array and calculates sector
        for every pixel location. If the 3*3 mask goes outside the edges of the image,
        the quantized value is set to nan
    """
    global gradient_angle, quantized_angle
    quantized_angle = np.zeros(gradient_magnitude.shape)
    for itr_x in range(quantized_angle.shape[0]):
        for itr_y in range(quantized_angle.shape[1]):
            if not np.isnan(gradient_angle[itr_x][itr_y]):
                quantized_angle[itr_x][itr_y] = get_sector(gradient_angle[itr_x][itr_y])
            else:
                quantized_angle[itr_x][itr_y] = np.nan
                
def quantize_angle_np(gradient_magnitude):
    """Function iterates over the gradient angle array and calculates sector
        for every pixel location. If the 3*3 mask goes outside the edges of the image,
        the quantized value is set to nan
    """
    global gradient_angle, quantized_angle_np
    quantized_angle_np = np.zeros(gradient_magnitude.shape)
    for itr_x in range(quantized_angle_np.shape[0]):
        for itr_y in range(quantized_angle_np.shape[1]):
            if not np.isnan(gradient_angle[itr_x][itr_y]):
                quantized_angle_np[itr_x][itr_y] = get_sector(gradient_angle[itr_x][itr_y])
            else:
                quantized_angle_np[itr_x][itr_y] = np.nan


def nms_compare(gradient_magnitude,ind_x, ind_y, sector):
    """Function calculates neighbor coordinates and checks if the center pixel is maximum out of the neighbors
        If the center pixel is the maximum, then function returns the magnitude of the pixel
        Else returns 0

    Args:
        ind_x (int): X coordinate of the center pixel
        ind_y (int): Y coordinate of the center pixel
        sector (int): Quantized gradient angle

    Returns:
        int: Gradient magnitude or 0
    """
    neighbor_l = {'x':ind_x+NEIGHBORS[sector]['l'][0],'y':ind_y+NEIGHBORS[sector]['l'][1]}
    neighbor_r = {'x':ind_x+NEIGHBORS[sector]['r'][0],'y':ind_y+NEIGHBORS[sector]['r'][1]}
    # print(f'\nsector: {sector}\ncx: {ind_x}, cy: {ind_y}\nnl{neighbor_l}, nr: {neighbor_r}')
    # print(f"compare: {gradient_magnitude[ind_x][ind_y]} > {gradient_magnitude[neighbor_l['x']][neighbor_l['y']]} and {gradient_magnitude[neighbor_r['x']][neighbor_r['y']]}")
    # print(f"compare res: {(gradient_magnitude[ind_x][ind_y] > gradient_magnitude[neighbor_l['x']][neighbor_l['y']]) and (gradient_magnitude[ind_x][ind_y] > gradient_magnitude[neighbor_r['x']][neighbor_r['y']])}\n")
    if (gradient_magnitude[ind_x][ind_y] >= gradient_magnitude[neighbor_l['x']][neighbor_l['y']]) and (gradient_magnitude[ind_x][ind_y] >= gradient_magnitude[neighbor_r['x']][neighbor_r['y']]):
        return gradient_magnitude[ind_x][ind_y]
    else:
        return 0
    
def nms(gradient_magnitude):
    """Non-maxima suppression function
        Steps:
            1) Quantize gradient angles
            2) Iterate over gradient magnitude and compare center pixel with neighbors
    """
    global quantized_angle
    magnitude_nms = np.zeros(gradient_magnitude.shape)
    for itr_x in range(magnitude_nms.shape[0]):
        for itr_y in range(magnitude_nms.shape[1]):
            if not np.isnan(quantized_angle[itr_x][itr_y]):
                magnitude_nms[itr_x][itr_y] = nms_compare(gradient_magnitude, itr_x, itr_y, quantized_angle[itr_x][itr_y])
            else:
                magnitude_nms[itr_x][itr_y] = 0
    return magnitude_nms

def nms_np(gradient_magnitude):
    """Non-maxima suppression function
        Steps:
            1) Quantize gradient angles
            2) Iterate over gradient magnitude and compare center pixel with neighbors
    """
    global quantized_angle_np
    magnitude_nms = np.zeros(gradient_magnitude.shape)
    start_x = start_y = 1
    for _ in range(magnitude_nms.shape[0]-2):
        for _ in range(magnitude_nms.shape[1]-2):
            # if not np.isnan(quantized_angle[itr_x][itr_y]):
            magnitude_nms[start_x][start_y] = nms_compare(gradient_magnitude, start_x, start_y, quantized_angle_np[start_x][start_y])
            start_y += 1
        start_x += 1
        start_y = 1
            # else:
            #     magnitude_nms[itr_x][itr_y] = 0
    return magnitude_nms
        
angle_calc(g,ix,iy)

print(f'ang: {gradient_angle.shape}')
print(f'pad:{pad}')

quantize_angle_np(g)

gradient_angle = np.pad(gradient_angle,pad,constant_values=(np.nan))
quantize_angle(g_pad)

# for i in range(gradient_angle.shape[0]):
#     for j in range(gradient_angle.shape[1]):
#         print(gradient_angle[i][j], quantized_angle[i][j])

g_nms = nms(g_pad)
print(f'g_nms: {g_nms.shape}')
g_nms_np = nms_np(g)
print(f'g_nms_np: {g_nms_np.shape}')

g_nms_np = np.pad(g_nms_np,pad)

print(f'after p g_nms_np: {g_nms_np.shape}')

print(f'gnms eq: {np.array_equal(g_nms_np, g_nms)}')

for i in range(g_nms.shape[0]):
    for j in range(g_nms.shape[1]):
        if g_nms[i][j] != g_nms_np[i][j]:
            print(i,j,g_nms[i][j],g_nms_np[i][j])

cv2.imwrite('gnms_np.bmp',g_nms_np)

threshold_25 = np.percentile(list(set(g_nms_np.flatten())),25)
threshold_50 = np.percentile(list(set(g_nms_np.flatten())),50)
threshold_75 = np.percentile(list(set(g_nms_np.flatten())),75)

print(threshold_25)
print(threshold_50)
print(threshold_75)