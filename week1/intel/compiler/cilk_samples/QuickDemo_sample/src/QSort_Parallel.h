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
// QSort_Parallel.h
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//

#pragma once

// Structure used to pass information to the parallel quicksort implementation

typedef struct
{
	__int64 iStart;			// Start time - filled in by serial sort thread
	HWND hWnd;				// HWND for dialog - recieves progress messages
	HANDLE hPause;			// Event handle to wait on
	volatile long lPause;	// Flag that we're to wait on the handle
	int n;					// Number of integers to sort
	int nProgressChunk;		// Threshold for when we send a message to the UI
	int iWorkers;			// Number of Intel(R) Cilk(TM) Plus workers to run
} parallel_qsort_info;

// Parallel quicksort thread procedure

extern unsigned __stdcall ParallelQSort (void *);
