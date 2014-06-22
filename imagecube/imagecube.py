# Licensed under a 3-clause BSD style license - see LICENSE.rst

# imagecube
# This package accepts FITS images from the user and delivers images that have
# been converted to the same flux units, registered to a common world 
# coordinate system (WCS), convolved to a common resolution, and resampled to a
# common pixel scale requesting the Nyquist sampling rate.
# Each step can be run separately or as a whole.
# The user should provide us with information regarding wavelength, pixel 
# scale extension of the cube, instrument, physical size of the target, and WCS
# header information.

from __future__ import print_function, division
from ..extern import six

import sys
import getopt
import glob
import math
import os
import warnings
import shutil

from datetime import datetime
from astropy import units as u
from astropy import constants
from astropy.io import fits
from astropy import wcs
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
import astropy.utils.console as console
import montage_wrapper as montage

import numpy as np
import scipy
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib import rc

NYQUIST_SAMPLING_RATE = 3.3
"""
Code constant: NYQUIST_SAMPLING_RATE

Some explanation of where this value comes from is needed.

"""

MJY_PER_SR_TO_JY_PER_ARCSEC2 = u.MJy.to(u.Jy)/u.sr.to(u.arcsec**2)
"""
Code constant: MJY_PER_SR_TO_JY_PER_ARCSEC2

Factor for converting Spitzer (MIPS and IRAC)  units from MJy/sr to
Jy/(arcsec^2)

"""

FUV_LAMBDA_CON = 1.40 * 10**(-15)
"""
Code constant: FUV_LAMBDA_CON

Calibration from CPS to Flux in [erg sec-1 cm-2 AA-1], as given in GALEX
for the FUV filter.
http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html

"""

NUV_LAMBDA_CON = 2.06 * 10**(-16)
"""
Code constant: NUV_LAMBDA_CON

calibration from CPS to Flux in [erg sec-1 cm-2 AA-1], as given in GALEX
for the NUV filter.
http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html

"""

FVEGA_J = 1594
"""
Code constant: FVEGA_J

Flux value (in Jy) of Vega for the 2MASS J filter.

"""

FVEGA_H = 1024
"""
Code constant: FVEGA_H

Flux value (in Jy) of Vega for the 2MASS H filter.

"""

FVEGA_KS = 666.7
"""
Code constant: FVEGA_KS

Flux value (in Jy) of Vega for the 2MASS Ks filter.

"""

WAVELENGTH_2MASS_J = 1.2409
"""
Code constant: WAVELENGTH_2MASS_J

Representative wavelength (in micron) for the 2MASS J filter

"""

WAVELENGTH_2MASS_H = 1.6514
"""
Code constant: WAVELENGTH_2MASS_H

Representative wavelength (in micron) for the 2MASS H filter

"""

WAVELENGTH_2MASS_KS = 2.1656
"""
Code constant: WAVELENGTH_2MASS_KS

Representative wavelength (in micron) for the 2MASS Ks filter

"""

JY_CONVERSION = u.Jy.to(u.erg / u.cm**2 / u.s / u.Hz, 1., 
                        equivalencies=u.spectral_density(u.AA, 1500))  ** -1
"""
Code constant: JY_CONVERSION

This is to convert the GALEX flux units given in erg/s/cm^2/Hz to Jy.

"""

S250_BEAM_AREA = 423
"""
Code constant: S250_BEAM_AREA

Beam area (arcsec^2) for SPIRE 250 band.
From SPIRE Observer's Manual v2.4.

"""
S350_BEAM_AREA = 751
"""
Code constant: S250_BEAM_AREA

Beam area (arcsec^2) for SPIRE 350 band.
From SPIRE Observer's Manual v2.4.

"""
S500_BEAM_AREA = 1587
"""
Code constant: S500_BEAM_AREA

Beam area (arcsec^2) for SPIRE 500 band.
From SPIRE Observer's Manual v2.4.

"""

def print_usage():
    """
    Displays usage information in case of a command line error.
    """

    print("""
Usage: """ + sys.argv[0] + """ --dir <directory> --ang_size <angular_size>
[--flux_conv] [--im_reg] [--im_ref <filename>] [--rot_angle <number in degrees>] 
[--im_conv] [--fwhm <fwhm value>] [--kernels <kernel directory>] [--im_regrid] 
[--im_pixsc <number in arcsec>] [--seds] [--cleanup] [--help]  

dir: the path to the directory containing the <input FITS files> to be 
processed. For multi-extension FITS files, currently only the first extension
after the primary one is used.

ang_size: the field of view of the output image cube in arcsec

flux_conv: perform unit conversion to Jy/pixel for all images not already
in these units.
NOTE: If data are not GALEX, 2MASS, MIPS, IRAC, PACS, SPIRE, then the user
should provide flux unit conversion factors to go from the image's native
flux units to Jy/pixel. This information should be recorded in the header
keyword FLUXCONV for each input image.

im_reg: register the input images to the reference image. The user should 
provide the reference image with the im_ref parameter.

im_ref: user-provided reference image to which the other images are registered. 
This image must have a valid world coordinate system. The position angle of
thie image will be used for the final registered images, unless an
angle is explicitly set using --rot_angle.

rot_angle: position angle (+y axis, in degrees West of North) for the registered images.
If omitted, the PA of the reference image is used.

im_conv: perform convolution to a common resolution, using either a Gaussian or
a PSF kernel. For Gaussian kernels, the angular resolution is specified with the fwhm 
parameter. If the PSF kernel is chosen, the user provides the PSF kernels with
the following naming convention:

    <input FITS files>_kernel.fits

For example: an input image named SI1.fits will have a corresponding
kernel file named SI1_kernel.fits

fwhm: the angular resolution in arcsec to which all images will be convolved with im_conv, 
if the Gaussian convolution is chosen, or if not all the input images have a corresponding kernel.

kernels: the name of a directory containing kernel FITS 
images for each of the input images. If all input images do not have a 
corresponding kernel image, then the Gaussian convolution will be performed for
these images.

im_regrid: perform regridding of the convolved images to a common
pixel scale. The pixel scale is defined by the im_pxsc parameter.

im_pixsc: the common pixel scale (in arcsec) used for the regridding
of the images in the im_regrid. It is a good idea the pixel scale and angular
resolution of the images in the regrid step to conform to the Nyquist sampling
rate: angular resolution = """ + `NYQUIST_SAMPLING_RATE` + """ * im_pixsc

seds:  produce the spectral energy distribution on a pixel-by-pixel
basis, on the regridded images.

cleanup: if this parameter is present, then output files from previous 
executions of the script are removed and no processing is done.

help: if this parameter is present, this message will be displayed and no 
processing will be done.

NOTE: the following keywords must be present in all images, along with a 
comment containing the units (where applicable), for optimal image processing:

    BUNIT: the physical units of the array values (i.e. the flux unit).
    FLSCALE: the factor that converts the native flux units (as given
             in the BUNIT keyword) to Jy/pixel. The units of this factor should
             be: (Jy/pixel) / (BUNIT unit). This keyword should be added in the
             case of data other than GALEX (FUV, NUV), 2MASS (J, H, Ks), 
             SPITZER (IRAC, MIPS), HERSCHEL (PACS, SPIRE; photometry)
    INSTRUME: the name of the instrument used
    WAVELNTH: the representative wavelength (in micrometres) of the filter 
              bandpass
Keywords which constitute a valid world coordinate system must also be present.

If any of these keywords are missing, imagecube will attempt to determine them.
The calculated values will be present in the headers of the output images; 
if they are not the desired values, please check the headers
of your input images and try again.
    """)


def parse_command_line():
    """
    Parses the command line to obtain parameters.

    """

    global ang_size
    global image_directory
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global main_reference_image
    global fwhm_input
    global kernel_directory
    global im_pixsc
    global rot_angle

##TODO: switch over to argparse
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["dir=", "ang_size=",
                                   "flux_conv", "im_conv", "im_reg", "im_ref=",
                                   "rot_angle=", "im_conv", "fwhm=", "kernels=", 
                                   "im_pixsc=","im_regrid", "seds", "cleanup", "help"])
    except getopt.GetoptError, exc:
        print(exc.msg)
        print("An error occurred. Check your parameters and try again.")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("--help"):
            print_usage()
            sys.exit()
        elif opt in ("--ang_size"):
            ang_size = float(arg)
        elif opt in ("--dir"):
            image_directory = arg
            if (not os.path.isdir(image_directory)):
                print("Error: The directory cannot be found: " + image_directory)
                sys.exit()
        elif opt in ("--flux_conv"):
            do_conversion = True
        elif opt in ("--im_reg"):
            do_registration = True
        elif opt in ("--rot_angle"):
            rot_angle = float(arg)
        elif opt in ("--im_conv"):
            do_convolution = True
        elif opt in ("--im_regrid"):
            do_resampling = True
        elif opt in ("--seds"):
            do_seds = True
        elif opt in ("--cleanup"):
            do_cleanup = True
        elif opt in ("--im_ref"):
            main_reference_image = arg
        elif opt in ("--fwhm"):
            fwhm_input = float(arg)
        elif opt in ("--kernels"):
            kernel_directory = arg
            if (not os.path.isdir(kernel_directory)):
                print("Error: The directory cannot be found: " + 
                      kernel_directory)
                sys.exit()
        elif opt in ("--im_pixsc"):
            im_pixsc = float(arg)

    if (main_reference_image != ''):
        try:
            with open(main_reference_image): pass
        except IOError:
            print("The file " + main_reference_image + 
                  " could not be found in the directory " + image_directory +
                  ". Cannot run without reference image, exiting.")
            sys.exit()
    return

def get_conversion_factor(header, instrument):
    """
    Returns the factor that is necessary to convert an image's native "flux 
    units" to Jy/pixel.

    Parameters
    ----------
    header: FITS file header
        The header of the FITS file to be checked.

    instrument: string
        The instrument which the data in the FITS file came from

    Returns
    -------
    conversion_factor: float
        The conversion factor that will convert the image's native "flux
        units" to Jy/pixel.
    """

    # Give a default value that can't possibly be valid; if this is still the
    # value after running through all of the possible cases, then an error has
    # occurred.
    conversion_factor = 0
    pixelscale = get_pixel_scale(header)

    if (instrument == 'IRAC'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'MIPS'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'GALEX'):
        wavelength = u.um.to(u.angstrom, float(header['WAVELNTH']))
        f_lambda_con = 0
        # I am using a < comparison here to account for the possibility that
        # the given wavelength is not EXACTLY 1520 AA or 2310 AA
        if (wavelength < 2000): 
            f_lambda_con = FUV_LAMBDA_CON
        else:
            f_lambda_con = NUV_LAMBDA_CON
        conversion_factor = (((JY_CONVERSION) * f_lambda_con * wavelength**2) /
                             (constants.c.to('angstrom/s').value))

    elif (instrument == '2MASS'):
        fvega = 0
        if (header['FILTER'] == 'j'):
            fvega = FVEGA_J
        elif (header['FILTER'] == 'h'):
            fvega = FVEGA_H
        elif (header['FILTER'] == 'k'):
            fvega = FVEGA_KS
        conversion_factor = fvega * 10**(-0.4 * header['MAGZP'])

    elif (instrument == 'PACS'):
        # Confirm that the data is already in Jy/pixel by checking the BUNIT 
        # header keyword
        if ('BUNIT' in header):
            if (header['BUNIT'].lower() != 'jy/pixel'):
                log.info("Instrument is PACS, but Jy/pixel is not being used in "
                      + "BUNIT.")
        conversion_factor = 1;

    elif (instrument == 'SPIRE'):
        wavelength = float(header['WAVELNTH'])
        if (wavelength == 250):
            conversion_factor = (pixelscale**2) / S250_BEAM_AREA
        elif (wavelength == 350):
            conversion_factor = (pixelscale**2) / S350_BEAM_AREA
        elif (wavelength == 500):
            conversion_factor = (pixelscale**2) / S500_BEAM_AREA
    
    return conversion_factor

def convert_images(images_with_headers):
    """
    Converts all of the input images' native "flux units" to Jy/pixel
    The converted values are stored in the list of arrays, 
    converted_data, and they are also saved as new FITS images.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/converted/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for i in range(0, len(images_with_headers)):
        if ('FLSCALE' in images_with_headers[i][1]):
            conversion_factor = float(images_with_headers[i][1]['FLSCALE'])
        else:
            try: # try to get conversion factor from image header
                instrument = images_with_headers[i][1]['INSTRUME']
                conversion_factor = get_conversion_factor(
                    images_with_headers[i][1], instrument)
            except KeyError: # get this if no 'INSTRUME' keyword
                conversion_factor = 0
            # if conversion_factor == 0 either we don't know the instrument
            # or we don't have a conversion factor for it.
            if conversion_factor == 0: 
                warnings.warn("No conversion factor for image %s, using 1"\
                     % images_with_headers[i][2],\
                    AstropyUserWarning)
                conversion_factor = 1.0

        # Some manipulation of filenames and directories
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        converted_filename = (new_directory + original_filename  + 
                              "_converted.fits")

        # Do a Jy/pixel unit conversion and save it as a new .fits file
        converted_data_array = images_with_headers[i][0] * conversion_factor
        converted_data.append(converted_data_array)
        images_with_headers[i][1]['BUNIT'] = 'Jy/pixel'
        images_with_headers[i][1]['JYPXFACT'] = (
            conversion_factor, 'Factor to'
            + ' convert original BUNIT into Jy/pixel.'
        )
        hdu = fits.PrimaryHDU(converted_data_array, images_with_headers[i][1])
        hdu.writeto(converted_filename, clobber=True)
    return

#modified from aplpy.wcs_util.get_pixel_scales
def get_pixel_scale(header):
    '''
    Compute the pixel scale in arcseconds per pixel from an image WCS
    Assumes WCS is in degrees (TODO: generalize)

    Parameters
    ----------
    header: FITS header of image


    '''
    w = wcs.WCS(header)
    # NB: get_cdelt is supposed to work whether header has CDij, PC, or CDELT
    pix_scale = abs(w.wcs.get_cdelt()[0]) * u.deg.to(u.arcsec)
    return(pix_scale)

def get_pangle(header):
    '''
    Compute the rotation angle, in degrees,  from an image WCS
    Assumes WCS is in degrees (TODO: generalize)

    Parameters
    ----------
    header: FITS header of image


    '''
#    try:
#        if 'CROTA2' in header.keys(): # use the CROTA2 kw if present
#            return(float(header['CROTA2']))
#        else: # otherwise use the CD matrix
#            cr2 = math.atan2(header['CD1_2'],header['CD2_2'])*u.radian.to(u.deg)
#            return(cr2) 
#    except KeyError:
#        warnings.warn('No PA information found!')
#        return(0.0)
    w = wcs.WCS(header)
    pc = w.wcs.get_pc()
    cr2 = math.atan2(pc[0,1],pc[0,0])*u.radian.to(u.deg)    
    return(cr2)

def merge_headers(montage_hfile, orig_header, out_file):
    '''
    Merges an original image header with the WCS info
    in a header file generated by montage.mHdr.
    Puts the results into out_file.


    Parameters
    ----------
    montage_hfile: a text file generated by montage.mHdr, 
    which contains only WCS information
    orig_header: FITS header of image, contains all the other
    stuff we want to keep

    '''
    montage_header = fits.Header.fromtextfile(montage_hfile)
    for key in orig_header.keys():
        if key in montage_header.keys():
            orig_header[key] = montage_header[key] # overwrite the original header WCS
    if 'CD1_1' in orig_header.keys(): # if original header has CD matrix instead of CDELTs:
        for cdm in ['CD1_1','CD1_2','CD2_1','CD2_2']: 
            del orig_header[cdm] # delete the CD matrix
        for cdp in ['CDELT1','CDELT2','CROTA2']: 
            orig_header[cdp] = montage_header[cdp] # insert the CDELTs and CROTA2
    orig_header.tofile(out_file,sep='\n',endcard=True,padding=False,clobber=True)
    return

def get_ref_wcs(img_name):
    '''
    get WCS parameters from first science extension
    (or primary extension if there is only one) of image

    Parameters
    ----------
    img_name: name of FITS image file


    '''
    hdulist = fits.open(img_name)
    hdr = hdulist[find_image_planes(hdulist)[0]].header #take the first sci image if multi-ext.
    lngref_input = hdr['CRVAL1']
    latref_input = hdr['CRVAL2']
    try:
        rotation_pa = rot_angle # the user-input PA
    except NameError: # user didn't define it
        log.info('Getting position angle from %s' % img_name)
        rotation_pa = get_pangle(hdr)
    log.info('Using PA of %.1f degrees' % rotation_pa)
    hdulist.close()
    return(lngref_input, latref_input, rotation_pa)

def find_image_planes(hdulist):
    """
    Reads FITS hdulist to figure out which ones contain science data

    Parameters
    ----------
    hdulist: FITS hdulist

    Outputs
    -------
    img_plns: list of which indices in hdulist correspond to science data

    """
    n_hdu = len(hdulist)
    img_plns = []
    if n_hdu == 1: # if there is only one extension, then use that
        img_plns.append(0)
    else: # loop over all the extensions & try to find the right ones
        for extn in range(1,n_hdu):
            try: # look for 'EXTNAME' keyword, see if it's 'SCI'
                if 'SCI' in hdulist[extn].header['EXTNAME']:
                    img_plns.append(extn)
            except KeyError: # no 'EXTNAME', just assume we want this extension
                img_plns.append(extn)
    return(img_plns)


def register_images(images_with_headers):
    """
    Registers all of the images to a common WCS

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/registered/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = get_ref_wcs(main_reference_image)
    width_and_height = u.arcsec.to(u.deg, ang_size)

    # now loop over all the images
    for i in range(0, len(images_with_headers)):

        native_pixelscale = get_pixel_scale(images_with_headers[i][1])

        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        artificial_filename = (new_directory + original_filename + 
                               "_pixelgrid_header")
        registered_filename = (new_directory + original_filename  + 
                               "_registered.fits")
        input_directory = original_directory + "/converted/"
        input_filename = (input_directory + original_filename  + 
                          "_converted.fits")

        # make the new header & merge it with old
        montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, 
                              width_and_height, artificial_filename, 
                              system='eq', equinox=2000.0, 
                              height=width_and_height, 
                              pix_size=native_pixelscale, rotation=rotation_pa)
        merge_headers(artificial_filename, images_with_headers[i][1], artificial_filename)
        # reproject using montage
        montage.wrappers.reproject(input_filename, registered_filename, 
                                   header=artificial_filename, exact_size=True)  
        # delete the file with header info
        os.unlink(artificial_filename)
    return

def convolve_images(images_with_headers):
    """
    Convolves all of the images to a common resolution using a simple
    gaussian kernel.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/convolved/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for i in range(0, len(images_with_headers)):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        convolved_filename = (new_directory + original_filename  + 
                              "_convolved.fits")
        input_directory = original_directory + "/registered/"
        input_filename = (input_directory + original_filename  + 
                          "_registered.fits")

        # Check if there is a corresponding PSF kernel.
        # If so, then use that to perform the convolution.
        # Otherwise, convolve with a Gaussian kernel.
        kernel_filename = (original_directory + "/" + kernel_directory + "/" + 
                           original_filename + "_kernel.fits")
        log.info("Looking for " + kernel_filename)

        if os.path.exists(kernel_filename):
            log.info("Found a kernel; will convolve with it shortly.")
            #reading the science image
            science_hdulist = fits.open(input_filename)
            science_header = science_hdulist[0].header
            science_image = science_hdulist[0].data
            science_hdulist.close()
            # reading the kernel
            kernel_hdulist = fits.open(kernel_filename)
            kernel_image = kernel_hdulist[0].data
            kernel_hdulist.close()
            # do the convolution and save as a new .fits file
            convolved_image = convolve_fft(science_image, kernel_image)
            hdu = fits.PrimaryHDU(convolved_image, science_header)
            hdu.writeto(convolved_filename, clobber=True)

        else: # no kernel
            native_pixelscale = get_pixel_scale(images_with_headers[i][1])
            sigma_input = (fwhm_input / 
                           (2* math.sqrt(2*math.log (2) ) * native_pixelscale))

            # NOTETOSELF: there has been a loss of data from the data cubes at
            # an earlier step. The presence of 'EXTEND' and 'DSETS___' keywords
            # in the header no longer means that there is any data in 
            # hdulist[1].data. I am using a workaround for now, but this needs
            # to be looked at.
            # NOTE_FROM_PB: can possibly solve this issue, and eliminate a lot 
            # of repetitive code, by making a multi-extension FITS file
            # in the initial step, and iterating over the extensions in that file
            hdulist = fits.open(input_filename)
            header = hdulist[0].header
            image_data = hdulist[0].data
            hdulist.close()
            # NOTETOSELF: not completely clear whether Gaussian2DKernel 'width' is sigma or FWHM
            # also, previous version had kernel being 3x3 pixels which seems pretty small!

            # construct kernel
            gaus_kernel_inp = Gaussian2DKernel(width=sigma_input)
            # Do the convolution and save it as a new .fits file
            conv_result = convolve(image_data, gaus_kernel_inp)
            header['FWHM'] = (fwhm_input, 
                              'FWHM value used in convolution, in pixels')
            hdu = fits.PrimaryHDU(conv_result, header)
            hdu.writeto(convolved_filename, clobber=True)
    return


def resample_images(images_with_headers, logfile_name):
    """
    Resamples all of the images to a common pixel grid.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/resampled/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # figure out the geometry of the resampled images
    width_input = ang_size / (im_pixsc) 
    height_input = width_input

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = get_ref_wcs(main_reference_image)

    # make the header for the resampled images (same for all)
    montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, width_input, 
                          'grid_final_resample_header', system='eq', 
                          equinox=2000.0, height=height_input, 
                          pix_size=im_pixsc, rotation=0.)

    for i in range(0, len(images_with_headers)):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        artificial_header = (new_directory + original_filename + 
                               "_artheader")
        resampled_filename = (new_directory + original_filename  + 
                              "_resampled.fits")
        input_directory = original_directory + "/convolved/"
        input_filename = (input_directory + original_filename  + 
                          "_convolved.fits")
        # generate header for regridded image
        merge_headers('grid_final_resample_header', images_with_headers[i][1],artificial_header)
        # do the regrid
        montage.wrappers.reproject(input_filename, resampled_filename, 
            header=artificial_header)  
        # delete the header file
        os.unlink(artificial_header)

    create_data_cube(images_with_headers, logfile_name, 'grid_final_resample_header')
    os.unlink('grid_final_resample_header')
    return

def create_data_cube(images_with_headers, logfile_name, header_text):
    """
    Creates a data cube from the provided images.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/datacube/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # make a new header
    prihdr = fits.Header()
    # put some information in it
    prihdr['CREATOR'] = 'IMAGECUBE'
    prihdr['DATE'] = datetime.now().strftime('%Y-%m-%d')
    prihdr['LOGFILE'] = logfile_name 
    if do_conversion:
        prihdr['BUNIT'] = ('Jy/pixel', 'Units of image data')
    # add the WCS info from the regridding stage
    # TODO: deal with case where this hasn't been done
    wcs_header = fits.Header.fromtextfile(header_text)
    for key in wcs_header.keys():
        if key not in prihdr.keys():
            prihdr[key] = wcs_header[key] 
    
    # now use this header to create a new fits file
    prihdu = fits.PrimaryHDU(header=prihdr)
    cube_hdulist = fits.HDUList([prihdu])

    for i in range(0, len(images_with_headers)):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        resampled_filename = (original_directory + "/resampled/" + 
                              original_filename  + "_resampled.fits")
        
        hdulist = fits.open(resampled_filename)
        cube_hdulist.append(hdulist[0])
        hdulist.close()

    cube_hdulist.writeto(new_directory + '/' + 'datacube.fits',clobber=True)
    return


def output_seds(images_with_headers):
    """
    Makes the SEDs.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/seds/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    all_image_data = []
    wavelengths = []

    num_wavelengths = len(images_with_headers)

    for i in range(0, num_wavelengths):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        input_directory = original_directory + "/resampled/"
        input_filename = (input_directory + original_filename  + 
                          "_resampled.fits")
        wavelength = images_with_headers[i][1]['WAVELNTH']
        wavelengths.append(wavelength)

        # Load the data for each image and append it to a master list of
        # all image data.
        ##NOTETOSELF: change to use nddata structure?
        hdulist = fits.open(input_filename)
        image_data = hdulist[0].data
        all_image_data.append(image_data)
        hdulist.close()

    sed_data = []
    for i in range(0, num_wavelengths):
        for j in range(len(all_image_data[i])):
            for k in range(len(all_image_data[i][j])):
                sed_data.append((int(j), int(k), wavelengths[i], 
                                all_image_data[i][j][k]))

    # write the SED data to a test file
    # NOTETOSELF: make this optional?
    data = np.copy(sorted(sed_data))
    np.savetxt('test.out', data, fmt='%f,%f,%f,%f', 
               header='x, y, wavelength (um), flux units (Jy/pixel)')
    num_seds = int(len(data) / num_wavelengths)

    with console.ProgressBarOrSpinner(num_seds, "Creating SEDs") as bar:
        for i in range(0, num_seds):

            # change to the desired fonts
            rc('font', family='Times New Roman')
            rc('text', usetex=True)
            # grab the data from the cube
            wavelength_values = data[:,2][i*num_wavelengths:(i+1)*
                                num_wavelengths]
            flux_values = data[:,3][i*num_wavelengths:(i+1)*num_wavelengths]
            # NOTETOSELF: change from 0-index to 1-index
            x_values = data[:,0][i*num_wavelengths:(i+1)*num_wavelengths] # pixel pos
            y_values = data[:,1][i*num_wavelengths:(i+1)*num_wavelengths] # pixel pos
            fig, ax = plt.subplots()
            ax.scatter(wavelength_values,flux_values)
            # axes specific
            ax.set_xlabel(r'Wavelength ($\mu$m)')					
            ax.set_ylabel(r'Flux density (Jy/pixel)')
            rc('axes', labelsize=14, linewidth=2, labelcolor='black')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(min(wavelength_values), max(wavelength_values)) #NOTETOSELF: doesn't quite seem to work
            ax.set_ylim(min(flux_values), max(flux_values))
            fig.savefig(new_directory + '/' + `int(x_values[0])` + '_' + 
                          `int(y_values[0])` + '_sed.eps')
            bar.update(i)
    return

def cleanup_output_files():
    """
    Removes files that have been generated by previous executions of the
    script.
    """

    for d in ('converted', 'registered', 'convolved', 'resampled', 'seds'):
        subdir = image_directory + '/' + d
        if (os.path.isdir(subdir)):
            log.info("Removing " + subdir)
            shutil.rmtree(subdir)


#if __name__ == '__main__':
def main(args=None):
    global ang_size
    global image_directory
    global main_reference_image
    global fwhm_input
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global kernel_directory
    global im_pixsc
    global rot_angle
    ang_size = ''
    image_directory = ''
    main_reference_image = ''
    fwhm_input = ''
    do_conversion = False
    do_registration = False
    do_convolution = False
    do_resampling = False
    do_seds = False
    do_cleanup = False
    kernel_directory = ''
    im_pixsc = ''

    parse_command_line()
    start_time = datetime.now()

    if (do_cleanup):
        cleanup_output_files()
        sys.exit()

    # if not just cleaning up, make a log file which records input parameters
    logfile_name = 'imagecube_'+ start_time.strftime('%Y-%m-%d_%H%M%S') + '.log'
    logf = open(logfile_name, 'w')
    logf.write(start_time.strftime('%Y-%m-%d_%H%M%S'))
    logf.write(': imagecube called with arguments %s' % sys.argv[1:])
    logf.close()

    # Grab all of the .fits and .fit files in the specified directory
    all_files = glob.glob(image_directory + "/*.fit*")
    # no use doing anything if there aren't any files!
    if len(all_files) == 0:
        warnings.warn('No fits found in directory' % image_directory, AstropyUserWarning )
        sys.exit()

    # Lists to store information
    global image_data
    global converted_data
    global registered_data
    global convolved_data
    global resampled_data
    global headers
    global filenames
    image_data = []
    converted_data = []
    registered_data = []
    convolved_data = []
    resampled_data = []
    headers = []
    filenames = []

    for (i,fitsfile) in enumerate(all_files):
        hdulist = fits.open(fitsfile)
        img_extens = find_image_planes(hdulist)
        # NOTETOSELF: right now we are just using the *first* image extension in a file
        #             which is not what we want to do, ultimately.
        header = hdulist[img_extens[0]].header
        image = hdulist[img_extens[0]].data
        # Strip the .fit or .fits extension from the filename so we can append
        # things to it later on
        filename = os.path.splitext(hdulist.filename())[0]
        hdulist.close()
        # check to see if image has reasonable scale & orientation
        # NOTETOSELF: should this really be here? It's not relevant for just flux conversion.
        #             want separate loop over image planes, after finishing file loop
        pixelscale = get_pixel_scale(header)
        fov = pixelscale * float(header['NAXIS1'])
        log.info("Checking %s: is pixel scale (%.2f\") < ang_size (%.2f\") < FOV (%.2f\") ?"% (fitsfile,pixelscale, ang_size,fov))
        if (pixelscale < ang_size < fov):
            try:
                wavelength = header['WAVELNTH'] 
                header['WAVELNTH'] = (wavelength, 'micron') # SOPHIA: why are we reading the keyword then setting it?
                image_data.append(image)
                headers.append(header)
                filenames.append(filename)
            except KeyError:
                warnings.warn('Image %s has no WAVELNTH keyword, will not be used' % filename, AstropyUserWarning)
        else:
            warnings.warn("Image %s does not meet the above criteria." % filename, AstropyUserWarning) 

    # Sort the lists by their WAVELNTH value
    images_with_headers_unsorted = zip(image_data, headers, filenames)
    images_with_headers = sorted(images_with_headers_unsorted, 
                                 key=lambda header: header[1]['WAVELNTH'])

    if (do_conversion):
        convert_images(images_with_headers)

    if (do_registration):
        register_images(images_with_headers)

    if (do_convolution):
        convolve_images(images_with_headers)

    if (do_resampling):
        resample_images(images_with_headers, logfile_name)

    if (do_seds):
        output_seds(images_with_headers)

    sys.exit()
