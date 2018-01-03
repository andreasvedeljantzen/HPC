/*

!==============================================================
!
! SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
! http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
!
! Copyright 2016 Intel Corporation
!
! THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
! NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
! PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
!
! =============================================================
*/
/*
    The original source for this example is
    Copyright (c) 1994-2008 John E. Stone
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. The name of the author may not be used to endorse or promote products
       derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
*/

/*
 *  jpeg.c - This file deals with JPEG format image files (reading/writing)
 *
 *  $Id: jpeg.cpp,v 1.2 2007-02-22 17:54:15  Exp $
 */ 

/*
 * This code requires support from the Independent JPEG Group's libjpeg.
 * For our puposes, we're interested only in the 3 byte per pixel 24 bit
 * RGB output.  Probably won't implement any decent checking at this point.
 */ 

#include <stdio.h>
#include "machine.h"
#include "types.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "jpeg.h"    /* the protos for this file */

#if !defined(USEJPEG)

int readjpeg(char * name, int * xres, int * yres, unsigned char **imgdata) {
  return IMAGEUNSUP;
}

#else

#include "jpeglib.h" /* the IJG jpeg library headers */

int readjpeg(char * name, int * xres, int * yres, unsigned char **imgdata) {
  FILE * ifp;
  struct jpeg_decompress_struct cinfo; /* JPEG decompression struct */
  struct jpeg_error_mgr jerr;          /* JPEG Error handler */
  JSAMPROW row_pointer[1];             /* output row buffer */
  int row_stride;                      /* physical row width in output buf */

  /* open input file before doing any JPEG decompression setup */
  if ((ifp = fopen(name, "rb")) == NULL) 
    return IMAGEBADFILE; /* Could not open image, return error */

  /*
   * Note: The Independent JPEG Group's library does not have a way
   *       of returning errors without the use of setjmp/longjmp.
   *       This is a problem in multi-threaded environment, since setjmp
   *       and longjmp are declared thread-unsafe by many vendors currently.
   *       For now, JPEG decompression errors will result in the "default"
   *       error handling provided by the JPEG library, which is an error
   *       message and a fatal call to exit().  I'll have to work around this
   *       or find a reasonably thread-safe way of doing setjmp/longjmp..
   */

  cinfo.err = jpeg_std_error(&jerr); /* Set JPEG error handler to default */

  jpeg_create_decompress(&cinfo);    /* Create decompression context      */ 
  jpeg_stdio_src(&cinfo, ifp);       /* Set input mechanism to stdio type */
  jpeg_read_header(&cinfo, TRUE);    /* Read the JPEG header for info     */
  jpeg_start_decompress(&cinfo);     /* Prepare for actual decompression  */

  *xres = cinfo.output_width;        /* set returned image width  */
  *yres = cinfo.output_height;       /* set returned image height */

  /* Calculate the size of a row in the image */
  row_stride = cinfo.output_width * cinfo.output_components;

  /* Allocate the image buffer which will be returned to the ray tracer */
  *imgdata = (unsigned char *) malloc(row_stride * cinfo.output_height);

  /* decompress the JPEG, one scanline at a time into the buffer */
  while (cinfo.output_scanline < cinfo.output_height) {
    row_pointer[0] = &((*imgdata)[(cinfo.output_scanline)*row_stride]);
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_decompress(&cinfo);   /* Tell the JPEG library to cleanup   */
  jpeg_destroy_decompress(&cinfo);  /* Destroy JPEG decompression context */

  fclose(ifp); /* Close the input file */

  return IMAGENOERR;  /* No fatal errors */
}

#endif
