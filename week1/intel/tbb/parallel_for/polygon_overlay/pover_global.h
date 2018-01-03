/*
    ==============================================================

    SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
    http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/

    Copyright 2016 Intel Corporation

    THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.

    =============================================================
*/

//
// pover_global.h
//
#ifndef _POVER_GLOBAL_H_
#define _POVER_GLOBAL_H_

#ifdef _MAIN_C_
#define DEFINE // nothing
#define STATIC static
#define INIT(n) = n
#else // not in main file
#define DEFINE extern
#define STATIC  // nothing
#define INIT(n) // nothing
#endif  // _MAIN_C_

#include <iostream>
#include <fstream>

#ifdef _WINDOWS
#include <windows.h>
#endif

// this Polygon class only supports rectangles
DEFINE int gDrawXOffset INIT(0);   // used for drawing polygons
DEFINE int gDrawYOffset INIT(0);
DEFINE int gPolyXBoxSize INIT(0);  // number of pixels orresponding to one "square" (x)
DEFINE int gPolyYBoxSize INIT(0);  // number of pixels orresponding to one "square" (y)
DEFINE bool gDoDraw INIT(false);       // render the boxes

#define THREADS_UNSET 0
DEFINE int gThreadsLow INIT(THREADS_UNSET);
DEFINE int gThreadsHigh INIT(THREADS_UNSET);

DEFINE std::ofstream gCsvFile;
DEFINE double gSerialTime;
DEFINE char *gCsvFilename INIT(NULL);

#define BORDER_SIZE 10  // number of pixels between maps

// The map size and the number of polygons depends on the version we are compiling.
// If DEBUG then it is small; else it is large.

#ifdef _DEBUG
DEFINE int gNPolygons INIT(30);  // default number of polygons in map
DEFINE int gMapXSize INIT(30);
DEFINE int gMapYSize INIT(30);
DEFINE int gGrainSize INIT(5);
#else
DEFINE int gNPolygons INIT(50000);    // default number of polygons in map
DEFINE int gMapXSize INIT(1000);
DEFINE int gMapYSize INIT(1000);
DEFINE int gGrainSize INIT(20);
#endif
DEFINE int gMyRandomSeed INIT(2453185);

DEFINE bool gIsGraphicalVersion INIT(false);

typedef enum {
    NORTH_SIDE,
    EAST_SIDE,
    SOUTH_SIDE,
    WEST_SIDE
} allSides;

#if _DEBUG
#define PRINT_DEBUG(x) (cout << x << std::endl)
#else
#define PRINT_DEBUG(x)
#endif


#endif // _POVER_GLOBAL_H_
