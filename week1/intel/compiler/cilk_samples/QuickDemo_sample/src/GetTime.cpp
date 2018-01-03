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
//
// GetTime.cpp
//
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// This file implements GetTime which gets the current time as a FileTime and
// returns it as an __int64 which we can do integer math on
//


#include "stdafx.h"
#include "GetTime.h"

__int64 GetTime()
{
	FILETIME ftNow;
	__int64 iNow;

	::GetSystemTimeAsFileTime (&ftNow);
	iNow = ((__int64)ftNow.dwHighDateTime << 32) + ftNow.dwLowDateTime;

	return iNow;
}