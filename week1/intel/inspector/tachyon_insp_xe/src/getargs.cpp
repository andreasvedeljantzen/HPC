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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "api.h"
#include "getargs.h"

void printusage(char **argv) {
  fprintf(stderr, "Usage: \n");
  fprintf(stderr, "  %s modelfile [options] \n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "Model file formats supported:\n");
  fprintf(stderr, "  filename.dat -- The model files originated with this package.\n");
  fprintf(stderr, "  filaname.ac  -- AC3D model files.\n");
  fprintf(stderr, "  filename.nff -- The NFF scene format used by Eric Haines' SPD.\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Valid options:  (** denotes default behaviour)\n");
  fprintf(stderr, " +D enable run-time display updating (if build supports it) **\n");
  fprintf(stderr, " -D disable run-time display updating\n");
  fprintf(stderr, " -nobounding\n");
  fprintf(stderr, " -boundthresh XXX  (** default threshold is 25)\n");
  fprintf(stderr, "\n");
}

void printnoparams(char **argv) {
  fprintf(stderr, "Using Default Image file\n");
  fprintf(stderr, " Use -? for additional options and usage model \n", argv[0]);
  fprintf(stderr, "\n");
}

void initoptions(argoptions * opt) {
  memset(opt, 0, sizeof(argoptions));
  opt->foundfilename = -1;
  opt->useoutfilename = -1;
  opt->verbosemode = -1;
  opt->antialiasing = -1;
  opt->displaymode = -1;
  opt->boundmode = -1; 
  opt->boundthresh = -1; 
  opt->usecamfile = -1;
}

int useoptions(argoptions * opt, SceneHandle scene) {
  if (opt->useoutfilename == 1) {
    rt_outputfile(scene, opt->outfilename);
  }

  if (opt->verbosemode == 1) {
    rt_verbose(scene, 1);
  }

  if (opt->antialiasing != -1) {
    /* need new api code for this */
  } 

  if (opt->displaymode != -1) {
    rt_displaymode(scene, opt->displaymode);
  }

  if (opt->boundmode != -1) {
    rt_boundmode(scene, opt->boundmode);
  }

  if (opt->boundthresh != -1) {
    rt_boundthresh(scene, opt->boundthresh);
  }

  return 0;
}    

int getparm(int argc, char **argv, int num, argoptions * opt) {
  if (!strcmp(argv[num], "-?")) {
	//printusage(argv);
    return -1;
  }  
  if (!strcmp(argv[num], "+D")) {
    /* turn video on */
    opt->displaymode = RT_DISPLAY_ENABLED;
    return 1;
  }
  if (!strcmp(argv[num], "-D")) {
    /* turn video off */
    opt->displaymode = RT_DISPLAY_DISABLED;
    return 1;
  }
  if (!strcmp(argv[num], "-nobounding")) {
    /* disable automatic spatial subdivision optimizations */
    opt->boundmode = RT_BOUNDING_DISABLED;
    return 1;
  }
  if (!strcmp(argv[num], "-boundthresh")) {
    /* set automatic bounding threshold control value */
    sscanf(argv[num + 1], "%d", &opt->boundthresh);
    return 2;
  }

  /* unknown parameter setting */
  fprintf(stderr, "Unrecognized parameter/option flag: %s\n", argv[num]);
  return -1;
}

int getargs(int argc, char **argv, argoptions * opt) {
  int i, rc, unknowncnt;

  if (opt == NULL)
    return -1;

  initoptions(opt);  

  if (argc < 2) {
    printnoparams(argv);
#ifndef DEFAULT_MODELFILE
    return -1;
#else
    return 0;
#endif//DEFAULT_MODELFILE
  }

  i = 1;
  unknowncnt = 0;
  while (i < argc) {
    if (argv[i][0] == '-' || argv[i][0] == '+') {
      rc = getparm(argc, argv, i, opt);
      if (rc != -1) {
        i += rc;
      }
      else {
        printusage(argv);
        return -1;
      }
    }
    else {
      unknowncnt++;
      if (unknowncnt > 1) {
        fprintf(stderr, "Too many model file names found!\n");
        printusage(argv); 
        return -1;
      } 
      else {
        strcpy(opt->filename, argv[i]);        
        opt->foundfilename = 1;
        i++;
      }
    }
  }

  if (opt->foundfilename == -1) {
    fprintf(stderr, "Missing model file name!\n");
    printusage(argv);
    return -1;
  }

  return 0;
}



