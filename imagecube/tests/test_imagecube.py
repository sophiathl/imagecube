# test script for imagecube
# modified from montage_wrappers/tests/test_wrappers.py
from __future__ import print_function, division

import os
import shutil
import tempfile
import warnings
from hashlib import md5

import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.tests.helper import pytest
from astropy.utils.exceptions import AstropyUserWarning

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

        # get the test data and copy it to the temp directory
        try:
            shutil.copytree('../data/testimgs',self.tmpdir+'/imagecubetest')
        except OSError:
            warnings.warn('cannot find test data') # change this to download


# end of class definition


# get rid of the temporary files
    def teardown_class(self):
        shutil.rmtree(self.tmpdir)
        return

# test the helper functions
    def test_helpers(self):
        pixscal_arcsec = imagecube.get_pixel_scale(self.header)
        assert_allclose(pixscal_arcsec/3600.0,cdelt_val)
        pa = imagecube.get_pangle(self.header)
        assert_allclose(pa,crota2_val)
        conv_fact1 = imagecube.get_conversion_factor(self.header,'MIPS') # assumed in MJy/sr
        assert_allclose(conv_fact1,u.MJy.to(u.Jy)/u.sr.to(u.arcsec**2) * (pixscal_arcsec**2))
        conv_fact2 = imagecube.get_conversion_factor(self.header,'BLINC') # unknown instrument, should give zero
        assert_allclose(conv_fact2,0.0)
        racen, deccen, crota = imagecube.get_ref_wcs('../data/I1_n5128_mosaic.fits') # eventually use astropy.utils.data
        assert racen == 201.243776
        assert deccen == -43.066428
        assert crota == 58.80616


# test the main imagecube script    
    def test_imagecube(self):
        # go where the test data are
        orig_dir = os.getcwd()
        os.chdir(self.tmpdir+'/imagecubetest')
        # run through the whole procedure
        # TBD: (or should we have a zipped list of images-with-headers, and test each step individually?)
        test_argstr = '--flux_conv --im_reg --im_conv --fwhm=8 --im_regrid --im_pixsc=3.0 --ang_size=300 --im_ref n5128_pbcd_24.fits --dir ./'  
        imagecube.main(args=test_argstr)

        # grab the output
        hdulist = fits.open(self.tmpdir+'/imagecubetest/datacube/datacube.fits')

        # check that we get the right shape output, with valid pixels
        assert hdulist[1].data.shape == (136,136)
        valid = hdulist[1].data[~np.isnan(hdulist[1].data)]
        assert len(valid) == 10284

        # compute and add the checksums        
        hdulist[1].add_datasum(when='testing')
        hdulist[1].add_checksum(when='testing',override_datasum=True)

        # test against values previously computed
        assert hdulist[1].header['DATASUM']== '3160818235' 
        assert hdulist[1].header['CHECKSUM']== 'ITDJJS9JISCJIS9J'
        hdulist.close()
        os.chdir(orig_dir)
        return

