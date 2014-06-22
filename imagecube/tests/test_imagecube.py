# test script for imagecube
# modified from montage_wrappers/tests/test_wrappers.py
from __future__ import print_function, division
from ..extern import six

import os
import shutil
import tempfile
from hashlib import md5

import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.tests.helper import pytest

from .. import imagecube

# Values for fake header input
# could randomize these to make a better test?
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

        # get the test data and copy it into the temp directory

# end of class definition


# get rid of the temporary files
    def teardown_class(self):
        shutil.rmtree(self.tmpdir)

# test the helper functions
    def test_helpers(self):
        pixscal_arcsec = imagecube.get_pixel_scale(self.header)
        pixscal_deg = round(pixscal_arcsec/3600.0,7) # rounding is not ideal
        assert pixscal_deg == cdelt_val
        pa = round(imagecube.get_pangle(self.header),1)
        assert pa == crota2_val
        conv_fact1 = imagecube.get_conversion_factor(self.header,'MIPS') # assumed in MJy/sr
        assert conv_fact1 == u.MJy.to(u.Jy)/u.sr.to(u.arcsec**2) * (pixscal_arcsec**2)
        conv_fact2 = imagecube.get_conversion_factor(self.header,'BLINC') # unknown instrument, should give zero
        assert conv_fact2 == 0.0
        racen, deccen, crota = imagecube.get_ref_wcs('I1_n5128_mosaic.fits')
        assert racen == 201.243776
        assert deccen == -43.066428
        assert crota == 58.80616

#    @pytest.mark.xfail()  # results are not consistent on different machines -- what does this do?

# test the main imagecube script    
#    def test_imagecube(self):
# run through the whole procedure
#        imagecube.__main__()  
#        # (or should we have a zipped list of images-with-headers, and test each step individually?)
# grab the output
#        hdulist = fits.open(tmpdir+'/datacube/datacube.fits')
# check that we get the right shape output, with valid pixels
#        assert hdulist[0].data.shape == (XX,YY)
#        valid = hdulist[0].data[~np.isnan(hdulist[0].data)]
#        assert len(valid) == 65029
# compute and add the checksums        
#         hdulist[0].add_datasum(when='testing')
#         hdulist[0].add_checksum(when='testing',override_datasum=True)
# test against values previously computed
#         assert hdulist[0].header['DATASUM']==dsum
#         assert hdulist[0].header['DATASUM']==csum
#         hdulist.close()
#
