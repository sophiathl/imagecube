# test script for imagecube
# modified from montage_wrappers/tests/test_wrappers.py
import os
import shutil
import tempfile
from hashlib import md5

import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS
from astropy.io import fits
from astropy.tests.helper import pytest

from .. import imagecube
#from imagecube import *

# Values for fake header input
cdelt_val = 0.0066667 # in degrees/pixel
crpix_val = 50.5 
cr1val_val = 10.5
cr2val_val = -43.0
crota2_val = 128.9

class TestImagecube(object):

    def setup_class(self):

        # make a fake header to test the helper functions which access the header
        w = WCS(naxis=2)

        w.wcs.crpix = [crpix_val, crpix_val]
        w.wcs.cdelt = np.array([-cdelt_val, cdelt_val])
        w.wcs.crval = [cr1val_val, cr2val_val]
        w.wcs.ctype = [b"RA---TAN", b"DEC--TAN"]
        w.wcs.crota = [0, crota2_val]

        self.header = w.to_header()

        # make a temporary directory for the input and output
        self.tmpdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tmpdir, 'raw'))

        # get the test data

# end of class definition


# get rid of the temporary files
    def teardown_class(self):
        shutil.rmtree(self.tmpdir)

    def test_helpers(self):
        pixscal_deg = round(imagecube.get_pixel_scale(self.header)/3600.0,7) # rounding is not ideal
        assert pixscal_deg == cdelt_val
        pa = round(imagecube.get_pangle(self.header),1)
        assert pa == crota2_val
        racen, deccen, crota = imagecube.get_ref_wcs(self.header)
        assert racen == crpix_val
        assert deccen == crpix_val
        assert crot == self.crota

#    @pytest.mark.xfail()  # results are not consistent on different machines -- what does this do?
#    def test_imagecube(self):
#        imagecube.__main__()  # run through the whole procedure
#        # or should we have a zipped list of images-with-headers, and test each step individually?
#        data_check = fits_checksum() # do a checksum on the resulting datacube?
#        header_check = fits_checksum() # do a checksum on the header
#        assert_allclose(header_check==)
#        assert_allclose(data_check==)
