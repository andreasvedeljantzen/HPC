
//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
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
 * parse.h - this file contains defines for model file reading.
 *

 */

#define PARSENOERR       0
#define PARSEBADFILE     1
#define PARSEBADSUBFILE  2
#define PARSEBADSYNTAX   4
#define PARSEEOF         8
#define PARSEALLOCERR    16
 
unsigned int readmodel(char *, SceneHandle);

#ifdef PARSE_INTERNAL
#define NUMTEXS 32768
#define TEXNAMELEN 24

typedef struct {
   double rx1; double rx2; double rx3;
   double ry1; double ry2; double ry3;
   double rz1; double rz2; double rz3;
} RotMat;

typedef struct {
        char name[TEXNAMELEN];
        void * tex;
} texentry;

#ifdef _ERRCODE_DEFINED
#define errcode errcode_t
#endif//_ERRCODE_DEFINED
typedef unsigned int errcode;

static errcode add_texture(void * tex, char name[TEXNAMELEN]);
static errcode GetString(FILE *, const char *);
static errcode GetScenedefs(FILE *, SceneHandle);
static errcode GetColor(FILE *, color *);
static errcode GetVector(FILE *, vector *);
static errcode GetTexDef(FILE *);
static errcode GetTexAlias(FILE *);
static errcode GetTexture(FILE *, void **);
void * GetTexBody(FILE *);
static errcode GetBackGnd(FILE *);
static errcode GetCylinder(FILE *);
static errcode GetFCylinder(FILE *);
static errcode GetPolyCylinder(FILE *);
static errcode GetSphere(FILE *);
static errcode GetPlane(FILE *);
static errcode GetRing(FILE *);
static errcode GetBox(FILE *);
static errcode GetVol(FILE *);
static errcode GetTri(FILE *);
static errcode GetSTri(FILE *);
static errcode GetLight(FILE *);
static errcode GetLandScape(FILE *);
static errcode GetTPolyFile(FILE *);
static errcode GetMGFFile(FILE *, SceneHandle);
static errcode GetObject(FILE *, SceneHandle);

#endif
