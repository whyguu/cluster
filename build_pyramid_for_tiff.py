import gdal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image")
flags = parser.parse_args()

ds = gdal.Open(flags.image)
ovlist = [2**a for a in range(1, 6)]
ds.BuildOverviews(overviewlist=ovlist)
