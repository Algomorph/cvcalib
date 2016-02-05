'''
Created on Jan 1, 2016

@author: Gregory Kramida
'''

import numpy as np

def generate_object_points(board_height, board_width, board_square_size):
    board_dims = (board_width,board_height)
    objp = np.zeros((board_height*board_width,1,3), np.float32)
    objp[:,:,:2] = np.indices(board_dims).T.reshape(-1, 1, 2)
    #convert square sizes to meters
    objp *= board_square_size
    return objp