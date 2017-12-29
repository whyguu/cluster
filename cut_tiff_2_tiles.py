#coding：utf-8
#Bin GAO

import os
from osgeo import gdal

# input path
in_path = '/Users/whyguu/Desktop/GF/0214/'
input_filename = 'GF2_PMS1_E124.9_N44.9_20170214_L1A0002186373-MSS1.tiff'

# output path
out_path = '/Users/whyguu/Desktop/GF/0214-cut/'
output_filename = 'jl_'

# patch size
tile_size_width = 500
tile_size_height = 500


ds = gdal.Open(in_path+input_filename)
band = ds.GetRasterBand(1)
x_size = band.XSize
y_size = band.YSize

for i in range(0, x_size, tile_size_width):
    for j in range(0, y_size, tile_size_height):
        cut = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " \
              + str(tile_size_width) + ", " + str(tile_size_height) + " " \
              + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) \
              + str(i) + "_" + str(j) + ".tif"
        os.system(cut)
