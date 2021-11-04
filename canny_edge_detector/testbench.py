import cv2
import numpy as np

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


si = np.pad(conv(img,gau),len(gau)//2)/140
si2 = convolution2(img, gau)/140

print(f'si: {si.shape}\nsi2: {si2.shape}')

for i in range(si.shape[0]):
    for j in range(si.shape[1]):
        if si[i][j] != si2[i][j]:
            print(f'i:{i}, j:{j}, si:{si[i][j]}, si2:{si2[i][j]}')

print(np.array_equal(si,si2))

cv2.imwrite('si.bmp',si)
cv2.imwrite('si2.bmp',si2)

ix = np.pad(conv(si,PREWITT_X),len(PREWITT_X)//2)

ix2 = convolution2(si2,PREWITT_X)

print(f'ix: {ix.shape}\nix2: {ix2.shape}')

for i in range(ix.shape[0]):
    for j in range(ix.shape[1]):
        if ix[i][j] != ix2[i][j]:
            print(f'i:{i}, j:{j}, ix:{ix[i][j]}, ix2:{ix2[i][j]}')

print(np.array_equal(ix,ix2))

cv2.imwrite('ix.bmp',ix)
cv2.imwrite('ix2.bmp',ix2)

iy = np.pad(conv(si,PREWITT_Y),len(PREWITT_Y)//2)

iy2 = convolution2(si2,PREWITT_Y)

print(f'iy: {iy.shape}\niy2: {iy2.shape}')

for i in range(iy.shape[0]):
    for j in range(iy.shape[1]):
        if iy[i][j] != iy2[i][j]:
            print(f'i:{i}, j:{j}, iy:{iy[i][j]}, iy2:{iy2[i][j]}')

print(np.array_equal(iy,iy2))

cv2.imwrite('iy.bmp',iy)
cv2.imwrite('iy2.bmp',iy2)

g = np.sqrt(np.square(ix)+np.square(iy))
g2 = np.sqrt(np.square(ix2)+np.square(iy2))

print(f'g: {g.shape}\ng2: {g2.shape}')

for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        if g[i][j] != g2[i][j]:
            print(f'i:{i}, j:{j}, g:{g[i][j]}, g2:{iy2[i][j]}')

print(np.array_equal(g,g2))

cv2.imwrite('g.bmp',g)
cv2.imwrite('g2.bmp',g2)