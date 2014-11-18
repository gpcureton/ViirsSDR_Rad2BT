#!/usr/bin/env python
# encoding: utf-8
"""
ViirsSDR_RadianceToBrightnessTemp.py

This script reads in the VIIRS VIIRS-SDR-EBBT-LUT file, and converts an
input VIIRS radiance into the corresponding brightness temperature.

Created by Geoff Cureton on 2010-10-17.
Copyright (c) 2010 University of Wisconsin SSEC. All rights reserved.
"""

file_Date = '$Date$'
file_Revision = '$Revision$'
file_Author = '$Author$'
file_HeadURL = '$HeadURL$'
file_Id = '$Id$'

__author__ = 'G.P. Cureton <geoff.cureton@ssec.wisc.edu>'
__version__ = '$Id$'
__docformat__ = 'Epytext'


import struct
import numpy as np
import scipy.weave as weave
from scipy.weave import converters

import adl_blob2


class ViirsRadToBtemp:

    def __init__(self,lutFile,xml_file=None,
        headers=[""],
        include_dirs=['.']):

        self.headers = headers
        self.include_dirs = include_dirs

        self.NUM_OF_EBBT_FILES = 7       # PRO/SDR/VIIRS/include/ProSdrViirsConstants.h:159
        self.MAX_EBBT_INDEX = 1050000    # PRO/SDR/VIIRS/include/SDR_Tables.h:163

        # PRO/SDR/VIIRS/include/SDR_Tables.h
        '''
        typedef struct
          {
            Int32    EBBT_indices[NUM_OF_EBBT_FILES][2];
            Float64 L_to_EBBT_tp[MAX_EBBT_INDEX];
            Float64 L_to_EBBT_rad[MAX_EBBT_INDEX];
          } proSdrViirsCalLtoEBBTLUT;

        proSdrViirsCalLtoEBBTLUT* EBBTLUT;
        '''

        ### Read data in LUT into local arrays

        if xml_file == None:
            f = open(lutFile, "rb")

            self.EBBT_indices = np.zeros((self.NUM_OF_EBBT_FILES,2),dtype=np.int32)

            for row in range(7):
                for col in range(2):
                    fourbytes = f.read(4)
                    self.EBBT_indices[row,col] = struct.unpack('i', fourbytes)[0]

            self.L_to_EBBT_tp = np.zeros(self.MAX_EBBT_INDEX,dtype=np.float64)

            for element in range(self.MAX_EBBT_INDEX):
                eightbytes = f.read(8)
                self.L_to_EBBT_tp[element] = struct.unpack('d', eightbytes)[0]

            self.L_to_EBBT_rad = np.zeros(self.MAX_EBBT_INDEX,dtype=np.float64)

            for element in range(self.MAX_EBBT_INDEX):
                eightbytes = f.read(8)
                self.L_to_EBBT_rad[element] = struct.unpack('d', eightbytes)[0]

            f.close()

        else:

            blobObj = adl_blob2.map(xml_file,lutFile,endian=adl_blob2.BIG_ENDIAN)
            self.EBBT_indices = np.array(getattr(blobObj,'EBBT_indices')).astype('int32')
            self.L_to_EBBT_rad = np.array(getattr(blobObj,'L_to_EBBT_rad')).astype('float64')
            self.L_to_EBBT_tp = np.array(getattr(blobObj,'L_to_EBBT_tp')).astype('float64')


        ### Compute some useful attributes for each channel
        self.minRadiance = []
        self.maxRadiance = []
        self.minBrightnessTemp = []
        self.maxBrightnessTemp = []

        for length,start in zip(self.EBBT_indices[:,0],self.EBBT_indices[:,1]):
            self.minRadiance.append(np.min(self.L_to_EBBT_rad[start:start+length]))
            self.maxRadiance.append(np.max(self.L_to_EBBT_rad[start:start+length]))
            self.minBrightnessTemp.append(np.min(self.L_to_EBBT_tp[start:start+length]))
            self.maxBrightnessTemp.append(np.max(self.L_to_EBBT_tp[start:start+length]))


    def interpLut(self,EBBT_indices,L_to_EBBT_rad,L_to_EBBT_tp,bandIndex,radiance):

        # From interpolate_L_to_EBBT() in PRO/SDR/VIIRS/Cal/src/InterpLUTs.cpp
        codeRadToBtemp = """
            /*
                IPO_1.5.0.48/trunk/INF/include/Typedefs.h
            */

            typedef char                Int8;
            typedef unsigned char       UInt8;
            typedef signed short int    Int16;
            typedef unsigned short int  UInt16;
            typedef int                 Int32;
            typedef unsigned int	    UInt32;
            typedef float               Float32;
            typedef double              Float64;

            /*
                IPO_1.5.0.48/trunk/PRO/include/ProCmnDefs.h
            */
            const Float64 NA_FLOAT64_FILL = -999.9e0;

            long idx=0;

            Int32 band_start = -1;
            Int32 array_start, array_count, low_index;
            Float64 delta_rad, rad_fraction;
            Float64 tp_interp=0.;

            band_start = bandIndex;

            /* store where in the two input arrays the data for the 
               band of interest starts (offset from beginning of array
               to start of data) */
            array_start = EBBT_indices[2 * band_start + 1];

            /* store the number of values in the band of interest */
            array_count = EBBT_indices[2 * band_start + 0];

            for (idx=0;idx<npts;idx++){
                Float64 rad = (Float64) radiance[idx];

                /* check to see if radiance to be operated on is out of range */
                if( (rad < *(L_to_EBBT_rad + array_start)) ||
                (rad > *(L_to_EBBT_rad + array_start + array_count - 1)) )
                {
                    tp_interp = NA_FLOAT64_FILL;
                }else{
                    /* inside the LUT the radiance values are evenly spaced,
                       therefore finding the delta radiance is one subtraction */
                    delta_rad = *(L_to_EBBT_rad + array_start + 1) - *(L_to_EBBT_rad + array_start);

                    /* calculate 1) the array index of the radiance closest to but less
                       than the radiance of interest and 2) the fraction used in
                       interpolation */
                    rad_fraction = (rad - *(L_to_EBBT_rad + array_start)) / delta_rad;
                    low_index = (Int32) rad_fraction;
                    rad_fraction -= low_index;

                    /* linear interpolation */
                    tp_interp = ( (1. - rad_fraction) *
                         *(L_to_EBBT_tp + array_start + low_index) ) +
                         ( rad_fraction *
                         *(L_to_EBBT_tp + array_start + low_index + 1) );
                }
                bTemp[idx] = tp_interp;
            }


            """

        # From interpolate_Temp_to_Rad() in PRO/SDR/VIIRS/Cal/src/InterpLUTs.cpp
        codeBtempToRad = """
            Int32 bandIndex = (Int32) radArr[0];
            Float64 tp =  (Float64) radArr[2];

            Int32 band_start = -1;
            Int32 array_start, array_count, low_index;
            Float64 L_to_EBBT_tp_Hi,L_to_EBBT_tp_Lo;
            Float64 delta_tp, tp_fraction;
            Float64 rad_interp=0.;

            band_start = bandIndex;

            printf("band_start = %d\\n",band_start);

            /* store where in the two input arrays the data for the 
            band of interest starts (offset from beginning of array
            to start of data) */
            array_start = EBBT_indices[2 * band_start + 1];

            printf("array_start = %d\\n",array_start);

            /* store the number of values in the band of interest */
            array_count = EBBT_indices[2 * band_start + 0];

            printf("array_count = %d\\n",array_count);

            /* check to see if brightness temperature to be operated on is out of range */
            if( (tp < *(L_to_EBBT_tp + array_start)) ||
            (tp > *(L_to_EBBT_tp + array_start + array_count - 1)) )
            {
                rad_interp = -1; //NA_FLOAT64_FILL;
            }
            printf("rad_interp = %f\\n",rad_interp);

            /* inside the LUT the temperature values are evenly spaced,
            therefore finding the delta temperature is one subtraction */
            L_to_EBBT_tp_Lo = *(L_to_EBBT_tp + array_start);
            L_to_EBBT_tp_Hi = *(L_to_EBBT_tp + array_start + 1);
            delta_tp = L_to_EBBT_tp_Hi - L_to_EBBT_tp_Lo;
            printf("delta_tp = %lf\\n",delta_tp);

            /* calculate 1) the array index of the temperature closest to but less
               than the temperature of interest and 2) the fraction used in
               interpolation */
            tp_fraction = (tp - *(L_to_EBBT_tp + array_start)) / delta_tp;
            printf("tp_fraction = %f\\n",tp_fraction);
            low_index = (Int32) tp_fraction;
            printf("low_index = %d\\n",low_index);
            tp_fraction -= low_index;
            printf("tp_fraction = %f\\n",tp_fraction);

            /* linear interpolation */
            rad_interp = ( (1. - tp_fraction) *
                *(L_to_EBBT_rad + array_start + low_index) ) +
                ( tp_fraction *
                *(L_to_EBBT_rad + array_start + low_index + 1) );

            printf("rad_interp = %f\\n",rad_interp);

            radArr[1] = rad_interp;

            """

        if (np.shape(radiance)==()):
            radiance = np.array(radiance,dtype=np.float64)
            bTemp = np.zeros((1),np.float64)
            npts = 1
        else :
            bTemp = np.zeros(np.shape(radiance),np.float64)
            npts = np.size(radiance) 

        weave.inline(codeRadToBtemp,
            arg_names=['EBBT_indices','L_to_EBBT_tp','L_to_EBBT_rad','bandIndex','npts','radiance','bTemp'],
            #include_dirs=self.include_dirs,
            #headers=self.headers,
            force=0)

        if (np.shape(radiance)==()):
            return bTemp[0]
        else :
            return bTemp


    def convertToBtemp(self,bandIndex,radiance):

        bTemp = 0.

        # Calculate the brightness temperature for this radiance
        bTemp = self.interpLut(self.EBBT_indices,self.L_to_EBBT_rad,self.L_to_EBBT_tp,bandIndex,radiance)

        return bTemp


    def convertToRadiance(self,bandIndex,bTemp):

        '''
        radiance = 0.

        # Calculate the radiance for this brightness temperature
        radiance = self.interpLut(self.EBBT_indices,
            self.L_to_EBBT_rad,self.L_to_EBBT_tp,bandIndex,bTemp)

        return radiance
        '''
        return ">>> convertToRadiance: Not yet implemented!"

if __name__=='__main__':
    lutFile   = str(sys.argv[1])
    bandIndex = int(sys.argv[2])
    radiance  = float(sys.argv[3])

    radToBtemp = ViirsRadToBtemp(lutFile)
    radToBtemp.convertToBtemp(bandIndex,radiance)

