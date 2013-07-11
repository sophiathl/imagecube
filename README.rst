The purpose of imagecube is to facilitate the process of assembling 
multi-wavelength astronomical imaging datasets. This critical intermediate step
between data discovery and scientific analysis involves a number of detailed
and tedious steps which can be automated by making use  of features in existing
Astropy modules.

It is common in astronomy to deal with observations of a target that are taken 
at different wavelengths of light, from X-rays to radio/submillimetre. These 
multi-wavelength observations, taken either from ground-based or from 
space-based telescopes, produce images with varying properties, including 
angular resolution, pixel scales,  and flux units. In order to perform 
multi-wavelength analyses, some common analysis steps have to be performed on 
each of a set of images. For example, the required image processing could 
involve conversion to standard flux units,  regridding to a specified pixel 
scale, and convolution to a specific angular resolution. Automating these tasks 
will free astronomers from repetitive calibration work and yield a 
multi-wavelength image cube ready for analysis.
