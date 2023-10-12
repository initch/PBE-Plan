import numpy as np
import copy

def add_pixel_pattern(data):
    
    new_data = copy.deepcopy(data)
    channels, height, width = new_data.shape
    for c in range(channels):
        new_data[c, height-3, width-3] = 255
        new_data[c, height-2, width-4] = 0
        new_data[c, height-4, width-2] = 255
        new_data[c, height-2, width-2] = 0

  
    return new_data
