#!/usr/bin/env python
# encoding: utf-8
"""
ViirsSDR_RadianceToBrightnessTemp_drv.py

This is a driver script for ViirsSDR_RadianceToBrightnessTemp.py

Created by Geoff Cureton on 2014-11-11.
Copyright (c) 2014 University of Wisconsin SSEC. All rights reserved.
"""

file_Date = '$Date$'
file_Revision = '$Revision$'
file_Author = '$Author$'
file_HeadURL = '$HeadURL$'
file_Id = '$Id$'

__author__ = 'G.P. Cureton <geoff.cureton@ssec.wisc.edu>'
__version__ = '$Id$'
__docformat__ = 'Epytext'


import os, sys
from os import path, uname, mkdir
from glob import glob
import string, logging, traceback
from time import time
from datetime import datetime

import numpy as np
from  numpy import ma as ma
import scipy as scipy

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Colormap,normalize,LinearSegmentedColormap,\
        ListedColormap,LogNorm
from matplotlib.figure import Figure

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# This must come *after* the backend is specified.
import matplotlib.pyplot as ppl

import h5py


# every module should have a LOG object
# e.g. LOG.warning('my dog has fleas')
import logging
LOG = logging.getLogger(__file__)

# Set up the logging
console_logFormat = '%(asctime)s : (%(levelname)s):%(filename)s:%(funcName)s:%(lineno)d:  %(message)s'
dateFormat = '%Y-%m-%d %H:%M:%S'
levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
#logging.basicConfig(level = levels[3], 
logging.basicConfig(level = logging.DEBUG, 
        format = console_logFormat, 
        datefmt = dateFormat)

import ViirsSDR_RadianceToBrightnessTemp


def SDR_histogram_generate(data_1,data_2, plotProd='BrightnessTemperature',
        histBins=20,histMin=None,histMax=None):


    # Construct fill masks to cover whatever isn't covered by
    # the bow-tie pixels.
    LOG.debug("Performing mask of float types")
    fill_mask_1 = ma.masked_less(data_1,-800.).mask
    fill_mask_2 = ma.masked_less(data_2,-800.).mask

    LOG.debug("data_1 is {}".format(data_1))
    LOG.debug("data_2 is {}".format(data_2))

    # Construct the total masks
    totalMask_1 = fill_mask_1
    totalMask_2 = fill_mask_1
    totalMask = np.ravel(totalMask_1 + totalMask_2)

    LOG.debug("totalMask_1.sum() = {}".format(totalMask_1.sum()))
    LOG.debug("totalMask_2.sum() = {}".format(totalMask_2.sum()))
    LOG.debug("totalMask.sum() = {}".format(totalMask.sum()))

    # Flatten the datasets
    data_1 = np.ravel(data_1)
    data_2 = np.ravel(data_2)
    LOG.debug("ravelled data_1.shape is {}".format(data_1.shape))
    LOG.debug("ravelled data_2.shape is {}".format(data_2.shape))

    # Mask the SDR so we only have the radiometric values
    data_1 = ma.masked_array(data_1,mask=totalMask)
    data_2 = ma.masked_array(data_2,mask=totalMask)

    LOG.debug("data_1.mask.sum() = {}".format(data_1.mask.sum()))
    LOG.debug("data_2.mask.sum() = {}".format(data_2.mask.sum()))

    ## Compress the datasets
    data_1 = ma.compressed(data_1)
    data_2 = ma.compressed(data_2)
    LOG.debug("compressed data_1.shape is {}".format(data_1.shape))
    LOG.debug("compressed data_2.shape is {}".format(data_2.shape))

    LOG.debug("data_1 is {}".format(data_1))
    LOG.debug("data_2 is {}".format(data_2))

    ## Generate the histogram for this granule

    LOG.info("Creating histogram...")

    vmin = np.min(data_1) if (histMin == None) else histMin
    vmax = np.max(data_1) if (histMax == None) else histMax
    LOG.debug("vmin is {}".format(vmin))
    LOG.debug("vmax is {}".format(vmax))

    histRange = np.array([[vmin,vmax],[vmin,vmax]])

    H, xedges, yedges = np.histogram2d(data_2,data_1,
            bins=histBins,range=histRange,normed=False)

    return H, xedges, yedges


def histogramPlot(xedges, yedges,histogram, 
        vmin=0.001,vmax=1.,histMin=None,histMax=None,scale=None,
        axis_label_1=None,axis_label_2=None,plot_title=r'',pngDpi=300, 
        cmap=None, pngName='SDR_hist.png'):
    '''
    Plots a 2D histogram defined by xedges, yedges, and histogram.
    '''

    figWidth = 5. # inches
    figHeight = 4.2 # inches

    # Create figure with default size, and create canvas to draw on
    fig = Figure(figsize=((figWidth,figHeight)))
    canvas = FigureCanvas(fig)

    ax_len = 0.80
    ax_x_len = ax_y_len = ax_len

    x0,y0 = 0.07,0.10
    x1,y1 = x0+ax_len,y0+ax_len

    cax_x_pad = 0.0
    cax_x_len = 0.05
    cax_y_len = ax_len

    ax_rect  = [x0, y0, ax_len , ax_len  ] # [left,bottom,width,height]
    cax_rect = [x1+cax_x_pad , y0, cax_x_len , cax_y_len ] # [left,bottom,width,height]

    LOG.info("ax_rect = {}".format(ax_rect))
    LOG.info("cax_rect = {}".format(cax_rect))

    timeString = 'Creation date: %s' %(datetime.strftime(datetime.utcnow(),"%Y-%m-%d %H:%M:%S Z"))
    fig.text(0.98, 0.01, timeString,fontsize=5, color='gray',ha='right',va='bottom',alpha=0.9)

    # the histogram axis ranges
    histMin = np.min(xedges) if (histMin == None) else histMin
    histMax = np.max(xedges) if (histMax == None) else histMax
    LOG.info("_histogramPlot Histogram range: {}".format([histMin, histMax]))

    # the histogram count range
    LOG.info("_histogramPlot Counts range: {}".format([vmin, vmax]))

    # Create main axes instance, leaving room for colorbar at bottom,
    # and also get the Bbox of the axes instance
    ax = fig.add_axes(ax_rect)
    Nbins = len(xedges) - 1
    parity = np.linspace(histMin,histMax,Nbins)
    parLine = ax.plot(parity,parity,'--')
    ppl.setp(parLine,color='gray')

    # Set the display ranges of the x and y axes...
    ax.set_xlim(histMin,histMax)
    ax.set_ylim(histMin,histMax)
    
    ppl.setp(ax.get_xticklabels(),fontsize=6)
    ppl.setp(ax.get_yticklabels(),fontsize=6)
    ppl.setp(ax,xlabel=axis_label_1)
    ppl.setp(ax,ylabel=axis_label_2)
    ax_title = ppl.setp(ax,title=plot_title)

    # The extent values just change the axis limits, they do NOT
    # alter which part of the array gets plotted.
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    LOG.info("xedges.shape = {}".format(xedges.shape))
    LOG.info("yedges.shape = {}".format(yedges.shape))

    H = histogram.astype(np.float)
    H /= np.max(H)
    LOG.info("Histogram.shape = {}".format(H.shape))
    LOG.info("Histogram min,max = {},{}".format(np.min(H),np.max(H)))

    LOG.info("Scaled Histogram min,max = {},{}".format(np.min(H),np.max(H)))

    cs = ax.imshow(H[:,:], extent=extent, vmin=vmin, vmax=vmax,
            interpolation='nearest',origin='lower',
            norm=LogNorm(vmin=vmin, vmax=vmax),cmap=cmap)

    # add a colorbar.
    cax = fig.add_axes(cax_rect,frameon=False) # setup colorbar axes

    t = [0.001,0.01,0.1,1.]
    cb = fig.colorbar(cs, cax=cax, ticks=t, format='$%.3f$', orientation='vertical')

    ppl.setp(cax.get_yticklabels(),fontsize=6)
    cax_title = ppl.setp(cax,title="counts/counts$_{max}$")
    ppl.setp(cax_title,fontsize=5)

    # Redraw the figure
    canvas.draw()

    # save image 
    LOG.info("Creating the image file {}".format((pngName)))
    canvas.print_figure(pngName,dpi=pngDpi)


def histogram2DPlot(xedges, yedges,histogram, 
        vmin=0.001,vmax=1.,histMin=None,histMax=None,scale=None,
        axis_label_1=None,axis_label_2=None,plot_title=r'',pngDpi=300, 
        cmap=None, pngName='SDR_hist.png'):
    '''
    Plots a 2D histogram defined by xedges, yedges, and histogram.
    '''

    figWidth = 5. # inches
    figHeight = 4.2 # inches

    # Create figure with default size, and create canvas to draw on
    fig = Figure(figsize=((figWidth,figHeight)))
    canvas = FigureCanvas(fig)

    ax_len = 0.80
    ax_x_len = ax_y_len = ax_len

    x0,y0 = 0.07,0.10
    x1,y1 = x0+ax_len,y0+ax_len

    cax_x_pad = 0.0
    cax_x_len = 0.05
    cax_y_len = ax_len

    ax_rect  = [x0, y0, ax_len , ax_len  ] # [left,bottom,width,height]
    cax_rect = [x1+cax_x_pad , y0, cax_x_len , cax_y_len ] # [left,bottom,width,height]

    LOG.info("ax_rect = {}".format(ax_rect))
    LOG.info("cax_rect = {}".format(cax_rect))

    timeString = 'Creation date: %s' %(datetime.strftime(datetime.utcnow(),"%Y-%m-%d %H:%M:%S Z"))
    fig.text(0.98, 0.01, timeString,fontsize=5, color='gray',ha='right',va='bottom',alpha=0.9)

    # the histogram axis ranges
    histMin = np.min(xedges) if (histMin == None) else histMin
    histMax = np.max(xedges) if (histMax == None) else histMax
    LOG.info("_histogramPlot Histogram range: {}".format([histMin, histMax]))

    # the histogram count range
    LOG.info("_histogramPlot Counts range: {}".format([vmin, vmax]))

    # Create main axes instance, leaving room for colorbar at bottom,
    # and also get the Bbox of the axes instance
    ax = fig.add_axes(ax_rect)
    Nbins = len(xedges) - 1
    parity = np.linspace(histMin,histMax,Nbins)
    parLine = ax.plot(parity,parity,'--')
    ppl.setp(parLine,color='gray')

    # Set the display ranges of the x and y axes...
    ax.set_xlim(histMin,histMax)
    ax.set_ylim(histMin,histMax)
    
    ppl.setp(ax.get_xticklabels(),fontsize=6)
    ppl.setp(ax.get_yticklabels(),fontsize=6)
    ppl.setp(ax,xlabel=axis_label_1)
    ppl.setp(ax,ylabel=axis_label_2)
    ax_title = ppl.setp(ax,title=plot_title)

    # The extent values just change the axis limits, they do NOT
    # alter which part of the array gets plotted.
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    LOG.info("xedges.shape = {}".format(xedges.shape))
    LOG.info("yedges.shape = {}".format(yedges.shape))

    H = histogram.astype(np.float)
    H /= np.max(H)
    LOG.info("Histogram.shape = {}".format(H.shape))
    LOG.info("Histogram min,max = {},{}".format(np.min(H),np.max(H)))

    LOG.info("Scaled Histogram min,max = {},{}".format(np.min(H),np.max(H)))

    cs = ax.imshow(H[:,:], extent=extent, vmin=vmin, vmax=vmax,
            interpolation='nearest',origin='lower',
            norm=LogNorm(vmin=vmin, vmax=vmax),cmap=cmap)

    # add a colorbar.
    cax = fig.add_axes(cax_rect,frameon=False) # setup colorbar axes

    t = [0.001,0.01,0.1,1.]
    cb = fig.colorbar(cs, cax=cax, ticks=t, format='$%.3f$', orientation='vertical')

    ppl.setp(cax.get_yticklabels(),fontsize=6)
    cax_title = ppl.setp(cax,title="counts/counts$_{max}$")
    ppl.setp(cax_title,fontsize=5)

    # Redraw the figure
    canvas.draw()

    # save image 
    LOG.info("Creating the image file {}".format((pngName)))
    canvas.print_figure(pngName,dpi=pngDpi)


def calc_bTemps(lut_file,xml_file=None):
    BT_prefixes = ['SVI04','SVI05','SVM12','SVM13','SVM14','SVM15','SVM16']
    bandIndices = np.arange(7)
    chan_dict = {}
    for prefix,bandIdx in zip(BT_prefixes,bandIndices):
        sdr_file = glob('{}*.h5'.format(prefix))[0]
        chan_dict[prefix] = {'sdr_file':sdr_file,'radiance':None,'bTemp':None,
                'bandIdx':bandIdx,'bTemp_new':None}

    # Initialise the ViirsRadToBtemp object with the LUT file...
    #lut_file = 'ViirsSDR_EBBT_1.5.0.48/proSdrViirsCalLtoEBBTLUT_le.bin'
    rad2Bt = ViirsSDR_RadianceToBrightnessTemp.ViirsRadToBtemp(
            lut_file,xml_file=xml_file)

    # Open the HDF5 file, and read the Radiance and RadianceFactors data.
    for prefix in BT_prefixes:
        # Open the SDR file
        sdr_file = chan_dict[prefix]['sdr_file']
        sdrObj = h5py.File(sdr_file,'r')

        # Read in the datasets
        band_key = '{}{}'.format(prefix[2],lstrip(prefix[3:],'0'))
        group_name = '/All_Data/VIIRS-{}-SDR_All'.format(band_key)
        print "Reading band {} datasets".format(group_name)

        radiance_str = '{}/Radiance'.format(group_name)
        bTemp_str = '{}/BrightnessTemperature'.format(group_name)
        radiance = sdrObj[radiance_str].value
        brightnessTemperature = sdrObj[bTemp_str].value

        try:
            radiance_str = '{}/RadianceFactors'.format(group_name)
            bTemp_str = '{}/BrightnessTemperatureFactors'.format(group_name)
            radianceFactors = sdrObj[radiance_str].value
            brightnessTemperatureFactors = sdrObj[bTemp_str].value

            # Determine where the bow-tie deleted pixels are, so we can restore
            # them with the type appropriate values after unscaling the
            # radiance...
            fill_mask = ma.masked_inside(radiance,65528,65535).mask

        except KeyError:
            radianceFactors = np.array([1.,0.])
            brightnessTemperatureFactors = np.array([1.,0.])
            fill_mask = ma.masked_inside(radiance,-999.2,-999.9).mask

        # Close the SDR file
        sdrObj.close()

        # Unscale the radiance, and restore the correct bow-tie fill value
        # for the float datatype...
        radiance = radiance*radianceFactors[0] + radianceFactors[1]
        radiance = ma.array(radiance,mask=fill_mask,fill_value=-999.3)
        radiance = radiance.filled().astype('float32')
        chan_dict[prefix]['radiance'] = radiance

        # Unscale the brightness temperature, and restore the correct bow-tie fill value
        # for the float datatype...
        brightnessTemperature = brightnessTemperature*brightnessTemperatureFactors[0] + brightnessTemperatureFactors[1]
        brightnessTemperature = ma.array(brightnessTemperature,mask=fill_mask,fill_value=-999.3)
        brightnessTemperature = brightnessTemperature.filled().astype('float32')
        chan_dict[prefix]['bTemp'] = brightnessTemperature

        # Convert the radiance to brightness temperature. The band indicies
        # are [0,1,..,6], corresponding to [I04,I05,M12,M13,M14,M15,M16].
        # Since we are doing M15, the band index is 5.
        bandIdx = chan_dict[prefix]['bandIdx']
        bTemp = rad2Bt.convertToBtemp(bandIdx,radiance)
        bTemp = ma.array(bTemp,mask=fill_mask,fill_value=-999.3)
        bTemp = bTemp.filled().astype('float32')
        chan_dict[prefix]['bTemp_new'] = bTemp


    return chan_dict


def plot_bTemps(chan_dict,chan_str,vmin=None,vmax=None):


    bTemp = ma.masked_less(chan_dict[chan_str]['bTemp'],-800.)

    f = ppl.figure(); 
    ppl.imshow(bTemp,interpolation='nearest',vmin=vmin,vmax=vmax); 
    ppl.colorbar(orientation='horizontal'); 
    ppl.title('IDPS {} Brightness Temperature (K)'.format(chan_str)); 
    ppl.show(block=False)

    bTemp_new = ma.masked_less(chan_dict[chan_str]['bTemp_new'],-800.)

    f = ppl.figure(); 
    ppl.imshow(bTemp_new,interpolation='nearest',vmin=vmin,vmax=vmax); 
    ppl.colorbar(orientation='horizontal'); 
    ppl.title('Converted IDPS {} Brightness Temperature (K)'.format(chan_str)); 
    ppl.show(block=False)


def plot_bTemp_diffs(chan_dict,chan_str,eps=1.):


    bTemp = ma.masked_less(chan_dict[chan_str]['bTemp'],-800.)
    bTemp_new = ma.masked_less(chan_dict[chan_str]['bTemp_new'],-800.)

    bTemp_diff = bTemp - bTemp_new
    #bTemp_diff = 100.*(bTemp - bTemp_new)/bTemp

    BT_diff_min = np.min(bTemp_diff)
    BT_diff_max = np.max(bTemp_diff)
    BT_diff_min_idx = np.where(bTemp_diff.data==BT_diff_min)
    BT_diff_max_idx = np.where(bTemp_diff.data==BT_diff_max)

    print "Minimum BT difference = {} @ {}".format(BT_diff_min,BT_diff_min_idx)
    print "Maximum BT difference = {} @ {}".format(BT_diff_max,BT_diff_max_idx)

    f = ppl.figure(); 
    ppl.imshow(bTemp_diff,interpolation='nearest',
            vmin=-1.*eps,vmax=eps,cmap='RdBu_r') 
    ppl.colorbar(orientation='horizontal'); 
    ppl.title('(Original - Converted) IDPS {} Brightness Temperature (K)'.format(chan_str)); 
    ppl.show(block=False)

