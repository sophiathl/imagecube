# adapted from montage_wrappers/tests/test_wrappers.py
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


class TestMosaic(object):

    def setup_class(self):

        # make a fake header to test the helper functions which access the header
        w = WCS(naxis=2)

        w.wcs.crpix = [50.5, 50.5]
        w.wcs.cdelt = np.array([-0.0066667, 0.0066667])
        w.wcs.crval = [lon[i], lat[j]]
        w.wcs.ctype = [b"RA---TAN", b"DEC--TAN"]
        w.wcs.crota = [0, np.random.uniform(0., 360.)]

        header = w.to_header()


        # make a temporary directory for the input and output
        self.tmpdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tmpdir, 'raw'))

        # get the test data

# end of class definition


# get rid of the temporary files
    def teardown_class(self):
        shutil.rmtree(self.tmpdir)

    def test_helpers(self):
        pixscal = get_pixel_scale(header)
        assert pixscal == XX
        pa = get_pangle(header)
        assert pa == YY
        racen, deccen, crota = get_ref_wcs(header)
        assert racen == RR
        assert deccen = DD
        assert crot = CC

#        valid = hdu.data[~np.isnan(hdu.data)]
#        assert len(valid) == 65029
#        assert_allclose(np.std(valid), 0.12658458001333581) # what does allclose do?
#        assert_allclose(np.mean(valid), 0.4995945318627074)
#        assert_allclose(np.median(valid), 0.5003376603126526)

    @pytest.mark.xfail()  # results are not consistent on different machines -- what does this do?
    def test_imagecube(self):
        imagecube.__main__()  # run through the whole procedure
        # or should we have a zipped list of images-with-headers, and test each step individually?
        data_check = fits_checksum() # do a checksum on the resulting datacube?
        header_check = fits_checksum() # do a checksum on the header
        assert_allclose(header_check==)
        assert_allclose(data_check==)
