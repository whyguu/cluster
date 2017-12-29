from osgeo import ogr, gdal
import subprocess


if __name__ == '__main__':
    InputVector = '/Users/whyguu/Desktop/临颍影像及地块数据/河南太保/line.shp'
    RefImage = '/Users/whyguu/Desktop/临颍影像及地块数据/google_河南太保_0p6meter.tif'
    OutputImage = 'label_raster.tif'

    gdalformat = 'GTiff'
    datatype = gdal.GDT_Byte
    burnVal = 1  # value for the output image pixels
    ##########################################################
    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    if not Shapefile:
        print('shapefile not open !')
        exit(0)
    if not Image:
        print('image not open !')
        exit(0)
    # Rasterise
    print("Rasterising shapefile...")
    Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype,
                                                     options=['COMPRESS=DEFLATE'])
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    print('starting write data ...')
    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    print('Band: ', Band)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

    # Close datasets
    Band = None
    Output = None
    Image = None
    Shapefile = None

    print('building image overviews...')
    # Build image overviews
    subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + OutputImage + " 2 4 8 16 32 64", shell=True)
    print("Done.")
