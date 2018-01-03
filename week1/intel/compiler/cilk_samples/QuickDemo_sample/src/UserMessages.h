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


//      User Message ID				Value				WParam				LParam

#define WMU_SERIAL_INITIALIZING		WM_USER	+ 100	//	unused				unused
#define WMU_SERIAL_STARTING			WM_USER + 101	//	unused				unused
#define WMU_SERIAL_BUMP_PROGRESS	WM_USER + 102	//	units completed		duration
#define WMU_SERIAL_DONE				WM_USER + 103	//	duration			unused

#define WMU_PARALLEL_INITIALIZING	WM_USER	+ 104	//	unused				unused
#define WMU_PARALLEL_STARTING		WM_USER + 105	//	unused				unused
#define WMU_PARALLEL_BUMP_PROGRESS	WM_USER + 106	//	units completed		duration
#define WMU_PARALLEL_DONE			WM_USER + 107	//	duration			unused
#define WMU_PARALLEL_COLOR_PROGRESS	WM_USER + 108
