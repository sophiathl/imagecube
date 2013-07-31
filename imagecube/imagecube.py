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

import sys
import getopt
import glob
import math
import os

from astropy import units as u
from astropy import constants
from astropy.io import fits
from astropy.nddata import make_kernel, convolve, convolve_fft
import astropy.utils.console as console
import montage_wrapper as montage

import numpy as np
import scipy
import pylab
from matplotlib import rc

NYQUIST_SAMPLING_RATE = 3.3
"""
Code constant: NYQUIST_SAMPLING_RATE

Some explanation of where this value comes from is needed.

"""

MJY_PER_SR_TO_JY_PER_ARCSEC2 = 2.3504 * 10**(-5)
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

Calibration from CPS to Flux in [erg sec-1 cm-2 AA-1], as given in GALEX
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

#JY_CONVERSION = 10**23
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
[--flux_conv] [--im_reg] [--im_ref <filename>] [--im_conv]
[--fwhm <fwhm value>] [--kernels] [--im_regrid] [--im_pixsc <number in arcsec>]
[--seds] [--cleanup] [--help]  

dir: the path to the directory containing the <input FITS files> to be 
processed

ang_size: the angular size of the object in arcsec

flux_conv: if flux units are not in Jy/pixel, this task will perform unit
conversion to Jy/pixel.
NOTE: If data are not GALEX, 2MASS, MIPS, IRAC, PACS, SPIRE, then the user
should provide flux unit conversion factors to go from the image's native
flux units to Jy/pixel. This information should be recorded in the header
keyword FLUXCONV for each input image.

im_reg: it performs the registration of the input images to the reference
image. The user should provide the reference image with the im_ref 
parameter.

im_ref: this is a reference image the user provides. In the header, the 
following keywords should be present: CRVAL1, CRVAL2, which give the RA and DEC
to which the images will be registered using im_reg.

im_conv: it performs convolution to a common resolution, either Gaussian
or using a PSF kernel. The user provides the angular resolution with the fwhm 
parameter. If the PSF kernel is chosen, the user provides the PSF kernels with
the following naming convention:

    <input FITS files>_kernel.fits

For example: an input image named SI1.fits will have a corresponding
kernel file named SI1_kernel.fits

fwhm: the user provides the angular resolution in arcsec to which all images
will be convolved with im_conv, if the Gaussian convolution is chosen, or if
not all the input images have a corresponding kernel.

kernels: the user provides kernel FITS images for each of the input images. If
all input images do not have a corresponding kernel image, then the Gaussian
convolution will be performed for these images.

im_regrid: it performs regridding of the convolved images to a common
pixel scale. The pixel scale is defined by the im_pxsc parameter.

im_pixsc: this gives the common pixel scale (in arcsec) used for the regridding
of the images in the im_regrid. It is a good idea the pixel scale and angular
resolution of the images in the regrid step to conform to the Nyquist sampling
rate: angular resolution = """ + `NYQUIST_SAMPLING_RATE` + """ * im_pixsc

seds: it produces the spectral energy distribution on a pixel-by-pixel
basis, on the regridded images.

cleanup: if this parameter is present, then output files from previous 
executions of the script are removed and no processing is done.

help: if this parameter is present, this message will be displayed and no 
processing will be done.

NOTE: the following keywords must be present in all images, along with a 
comment containing the units (where applicable), for optimal image processing:

    BUNIT: this provides the physical units of the array values (i.e. the flux
           unit).
    CRVAL1: it contains the RA (in degrees) to which the images will be 
            registered by im_reg
    CRVAL2: it contains the DEC (in degrees) to which the images will be 
            registered by im_reg
    CDELT1: the pixelscale (in degrees) along the x-axis
    CDELT2: the pixelscale (in degrees) along the y-axis
    FLSCALE: this is the factor that converts the native flux units (as given
             in the BUNIT keyword) to Jy/pixel. The units of this factor should
             be: (Jy/pixel) / (BUNIT unit). This keyword should be added in the
             case of data other than GALEX (FUV, NUV), 2MASS (J, H, Ks), 
             SPITZER (IRAC, MIPS), HERSCHEL (PACS, SPIRE; photometry)
    INSTRUME: this provides the instrument information
    WAVELNTH: the representative wavelength (in micrometres) of the filter 
              bandpass

If any of these keywords are missing, imagecube will attempt to determine them 
as best as possible. The calculated values will be present in the headers of 
the output images; if they are not the desired values, please check the headers
of your input images and make sure that these values are present.
    """)

def parse_command_line():
    """
    Parses the command line to obtain parameters.

    """

    global ang_size
    global directory
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global main_reference_image
    global fwhm_input
    global use_kernels
    global im_pixsc

    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["dir=", "ang_size=",
                                   "flux_conv", "im_conv", "im_reg", "im_ref=",
                                   "im_conv", "fwhm=", "kernels", "im_pixsc=",
                                   "im_regrid", "seds", "cleanup", "help"])
    except getopt.GetoptError:
        print("An error occurred. Check your parameters and try again.")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("--help"):
            print_usage()
            sys.exit()
        elif opt in ("--ang_size"):
            ang_size = float(arg)
        elif opt in ("--dir"):
            directory = arg
            if (not os.path.isdir(directory)):
                print("Error: The directory cannot be found: " + directory)
                sys.exit()
        elif opt in ("--flux_conv"):
            do_conversion = True
        elif opt in ("--im_reg"):
            do_registration = True
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
            use_kernels = True
        elif opt in ("--im_pixsc"):
            im_pixsc = float(arg)

    if (main_reference_image != ''):
        try:
            with open(main_reference_image): pass
        except IOError:
            print("The file " + main_reference_image + 
                  " could not be found in the directory " + directory)
            sys.exit()

def check_ref_image_pixel_scale(ang_size, main_reference_image):
    """
    Compares the pixel scale of the reference image to the angular size
    provided by the user. If the former is smaller than the latter, an
    error message is displayed and the script exits.

    Parameters
    ----------
    ang_size: float
        The angular size provided by the user with the ang_size parameter.

    main_reference_image: name of a FITS file
        The reference image provided by the user with the im_ref parameter.
    """
    pixelscale = u.deg.to(
        u.arcsec, abs(float(fits.getval(main_reference_image, 'CDELT1')))
    )
    print("Comparing " + `pixelscale` + " to " + `ang_size`)

    if (ang_size < pixelscale):
        print("Angular size is smaller than pixel scale of reference image.")
        print("Please check your parameters and try again.")
        sys.exit()


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
    pixelscale = u.deg.to(u.arcsec, abs(float(header['CDELT1'])))


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
        # header# keyword
        if ('BUNIT' in header):
            if (header['BUNIT'].lower() != 'jy/pixel'):
                print("Instrument is PACS, but Jy/pixel is not being used in "
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

    for i in range(0, len(images_with_headers)):
        if ('FLSCALE' in images_with_headers[i][1]):
            conversion_factor = float(images_with_headers[i][1]['FLSCALE'])
        else:
            instrument = images_with_headers[i][1]['INSTRUME']
            conversion_factor = get_conversion_factor(
                images_with_headers[i][1], instrument
            )

        # Some manipulation of filenames and directories
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        new_directory = original_directory + "/converted/"
        converted_filename = (new_directory + original_filename  + 
                              "_converted.fits")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

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

def register_images(images_with_headers):
    """
    Registers all of the images to a common WCS

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """

    hdr = fits.getheader(main_reference_image, 0)
    lngref_input = hdr['CRVAL1']
    latref_input = hdr['CRVAL2']
    width_and_height = u.arcsec.to(u.deg, ang_size)

    for i in range(0, len(images_with_headers)):

        native_pixelscale = u.deg.to(u.arcsec, 
            abs(float(images_with_headers[i][1]['CDELT1'])))

        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        new_directory = original_directory + "/registered/"
        artificial_filename = (new_directory + original_filename + 
                               "_pixelgrid_header")
        registered_filename = (new_directory + original_filename  + 
                               "_registered.fits")
        input_directory = original_directory + "/converted/"
        input_filename = (input_directory + original_filename  + 
                          "_converted.fits")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, 
                              width_and_height, artificial_filename, 
                              system='eq', equinox=2000.0, 
                              height=width_and_height, 
                              pix_size=native_pixelscale, rotation=0.)
        montage.wrappers.reproject(input_filename, registered_filename, 
                                   header=artificial_filename, exact_size=True)  
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

    for i in range(0, len(images_with_headers)):

        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        new_directory = original_directory + "/convolved/"
        convolved_filename = (new_directory + original_filename  + 
                              "_convolved.fits")
        input_directory = original_directory + "/registered/"
        input_filename = (input_directory + original_filename  + 
                          "_registered.fits")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        # Check if there is a corresponding PSF kernel.
        # If so, then use that to perform the convolution.
        # Otherwise, we convolve with a Gaussian kernel.
        kernel_filename = (original_directory + "/kernels/" + original_filename
                           + "_kernel.fits")
        print("Looking for " + kernel_filename)

        if use_kernels and os.path.exists(kernel_filename):

            print("Found a kernel; will convolve with it shortly.")
            #reading the science image:
            #science_image = fits.getdata(input_filename)
            science_hdulist = fits.open(input_filename)
            science_header = science_hdulist[0].header
            science_image = science_hdulist[0].data
            science_hdulist.close()
            # reading the kernel
            #kernel_image = fits.getdata(kernel_filename)
            kernel_hdulist = fits.open(kernel_filename)
            kernel_image = kernel_hdulist[0].data
            kernel_hdulist.close()

            convolved_image = convolve_fft(science_image, kernel_image)
            hdu = fits.PrimaryHDU(convolved_image, science_header)
            hdu.writeto(convolved_filename, clobber=True)
            #fits.writeto(convolved_filename, convolved_image, clobber=True)

        else:
            native_pixelscale = u.deg.to(
                u.arcsec, abs(float(images_with_headers[i][1]['CDELT1']))
            )
            sigma_input = (fwhm_input / 
                           (2* math.sqrt(2*math.log (2) ) * native_pixelscale))

            # NOTETOSELF: there has been a loss of data from the data cubes at
            # an earlier step. The presence of 'EXTEND' and 'DSETS___' keywords
            # in the header no longer means that there is any data in 
            # hdulist[1].data. I am using a workaround for now, but this needs
            # to be looked at.
            hdulist = fits.open(input_filename)
            header = hdulist[0].header
            image_data = hdulist[0].data
            hdulist.close()

            gaus_kernel_inp = make_kernel([3,3], kernelwidth=sigma_input, 
                                          kerneltype='gaussian', 
                                          trapslope=None, force_odd=True)

            # Do the convolution and save it as a new .fits file
            conv_result = convolve(image_data, gaus_kernel_inp)
            header['FWHM'] = (fwhm_input, 
                              'The FWHM value used in the convolution step.')

            hdu = fits.PrimaryHDU(conv_result, header)
            hdu.writeto(convolved_filename, clobber=True)

def create_data_cube(images_with_headers):
    """
    Creates a data cube from the provided images.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    Notes
    -----
    Currently we are just using the header of the first input image.
    This should be changed to something more appropriate.
    """
    resampled_images = []
    resampled_headers = []

    new_directory = directory + "/datacube/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for i in range(0, len(images_with_headers)):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        resampled_filename = (original_directory + "/resampled/" + 
                              original_filename  + "_resampled.fits")

        hdulist = fits.open(resampled_filename)
        header = hdulist[0].header
        resampled_headers.append(header)
        image = hdulist[0].data
        resampled_images.append(image)
        hdulist.close()

    fits.writeto(new_directory + '/' + 'datacube.fits', np.copy(resampled_images), resampled_headers[0], clobber=True)

def resample_images(images_with_headers):
    """
    Resamples all of the images to a common pixel grid.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """

    width_input = ang_size / (im_pixsc) 
    height_input = width_input

    hdr = fits.getheader(main_reference_image, 0)
    lngref_input = hdr['CRVAL1']
    latref_input = hdr['CRVAL2']

    montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, width_input, 
                          'grid_final_resample_header', system='eq', 
                          equinox=2000.0, height=height_input, 
                          pix_size=im_pixsc, rotation=0.)

    for i in range(0, len(images_with_headers)):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        new_directory = original_directory + "/resampled/"
        resampled_filename = (new_directory + original_filename  + 
                              "_resampled.fits")
        input_directory = original_directory + "/convolved/"
        input_filename = (input_directory + original_filename  + 
                          "_convolved.fits")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        montage.wrappers.reproject(input_filename, resampled_filename, 
            header='grid_final_resample_header')  

    create_data_cube(images_with_headers)

def output_seds(images_with_headers):
    """
    Makes the SEDs.

    Parameters
    ----------
    images_with_headers: zipped list structure
        A structure containing headers and image data for all FITS input
        images.

    """

    all_image_data = []
    wavelengths = []

    num_wavelengths = len(images_with_headers)

    for i in range(0, num_wavelengths):
        original_filename = os.path.basename(images_with_headers[i][2])
        original_directory = os.path.dirname(images_with_headers[i][2])
        new_directory = original_directory + "/seds/"
        input_directory = original_directory + "/resampled/"
        input_filename = (input_directory + original_filename  + 
                          "_resampled.fits")
        wavelength = images_with_headers[i][1]['WAVELNTH']
        wavelengths.append(wavelength)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        # Load the data for each image and append it to a master list of
        # all image data.
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

    data = np.copy(sorted(sed_data))
    np.savetxt('test.out', data, fmt='%d,%d,%f,%f', 
               header='x, y, wavelength (um), flux units (Jy/pixel)')
    num_seds = int(len(data) / num_wavelengths)

    with console.ProgressBarOrSpinner(num_seds, "Creating SEDs") as bar:
        for i in range(0, num_seds):

            # change to the desired fonts
            rc('font', family='Times New Roman')
            rc('text', usetex=True)
            
            wavelength_values = data[:,2][i*num_wavelengths:(i+1)*
                                num_wavelengths]
            flux_values = data[:,3][i*num_wavelengths:(i+1)*num_wavelengths]
            x_values = data[:,0][i*num_wavelengths:(i+1)*num_wavelengths]
            y_values = data[:,1][i*num_wavelengths:(i+1)*num_wavelengths]

            pylab.figure(i)
            pylab.scatter(wavelength_values,flux_values)

            # axes specific
            pylab.xlabel(r'log(Wavelength) (um)')					
            pylab.ylabel(r'Flux (Jy/pixel)')
            pylab.rc('axes', labelsize=14, linewidth=2, labelcolor='black')
            pylab.semilogx()
            pylab.axis([min(wavelength_values), max(wavelength_values),
                       min(flux_values), max(flux_values)])

            pylab.hold(True)

            pylab.legend()
            pylab.savefig(new_directory + '/' + `int(x_values[0])` + '_' + 
                          `int(y_values[0])` + '_sed.eps')
            #pylab.show()
            bar.update(i)

def cleanup_output_files():
    """
    Removes files that have been generated by previous executions of the
    script.
    """

    import shutil

    for d in ('converted', 'registered', 'convolved', 'resampled', 'seds'):
        subdir = directory + '/' + d
        if (os.path.isdir(subdir)):
            print("Removing " + subdir)
            shutil.rmtree(subdir)

#if __name__ == '__main__':
def main(args=None):
    global ang_size
    global directory
    global main_reference_image
    global fwhm_input
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global use_kernels
    global im_pixsc
    ang_size = ''
    directory = ''
    main_reference_image = ''
    fwhm_input = ''
    do_conversion = False
    do_registration = False
    do_convolution = False
    do_resampling = False
    do_seds = False
    do_cleanup = False
    use_kernels = False
    im_pixsc = ''

    parse_command_line()

    if (do_cleanup):
        cleanup_output_files()
        sys.exit()

    check_ref_image_pixel_scale(ang_size, main_reference_image)

    # Grab all of the .fits and .fit files in the specified directory
    all_files = glob.glob(directory + "/*.fit*")

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

    for i in all_files:
        hdulist = fits.open(i)
        header = hdulist[0].header
        # NOTETOSELF: The check for a data cube needs to be another function
        # due to complexity. Check the hdulist.info() values to see how much 
        # information is contained in the file.  In a data cube, there may be 
        # more than one usable science image. We need to make sure that they 
        # are all grabbed.
        # Check to see if the input file is a data cube before trying to grab 
        # the image data.
        if ('EXTEND' in header and 'DSETS___' in header):
            image = hdulist[1].data
        else:
            image = hdulist[0].data
        # Strip the .fit or .fits extension from the filename so we can append
        # things to it later on
        filename = os.path.splitext(hdulist.filename())[0]
        hdulist.close()
        wavelength = header['WAVELNTH']
        header['WAVELNTH'] = (wavelength, 'micron')
        image_data.append(image)
        headers.append(header)
        filenames.append(filename)

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
        resample_images(images_with_headers)

    if (do_seds):
        output_seds(images_with_headers)

    sys.exit()
