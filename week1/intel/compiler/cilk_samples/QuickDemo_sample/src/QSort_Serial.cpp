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
// QSort_Serial.cpp
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// This file implements a serial quicksort.
//

#include "stdafx.h"
#include <algorithm>
#include <iterator>
#include <functional>
#include <math.h>

#include "QSort_Serial.h"
#include "GetTime.h"
#include "UserMessages.h"

using namespace std;

//
// qsort_inner
// 
// Simple serial implementation of QSort.  The inner function doesn't have
// to pass around information for the messages to the UI.
//

static
void qsort_inner (serial_qsort_info *pSortInfo, int *begin, int *end)
{
    // Check wheter we're paused.  We use both a global variable and an
    // event to avoid constant kernel transitions to check the event.

    if (pSortInfo->lPause)
        WaitForSingleObject (pSortInfo->hPause, INFINITE);

    // Do the quicksort

    if (begin != end)
    {
        --end;  // Exclude last element (pivot) from partition

        int *middle = partition (begin, end, bind2nd (less<typename iterator_traits<int *>::value_type>(), *begin));

        using std::swap;
        swap(*end, *middle);                        // Move pivot to middle

        qsort_inner(pSortInfo, begin, middle);
        qsort_inner(pSortInfo, ++middle, ++end);    // Exclude pivot and restore end
    }
}

//
// qsort
//
// Simple serial implementation of QSort.  If we get below nProgressChunk,
// switch to the inner function and post a message to the dialog when that
// chunk of work is done.
//

static
void qsort (serial_qsort_info *pSortInfo, int *begin, int *end)
{
    // Check wheter we're paused.  We use both a global variable and an
    // event to avoid constant kernel transitions to check the event.

    if (pSortInfo->lPause)
        WaitForSingleObject (pSortInfo->hPause, INFINITE);

    // If we've dropped below the threshold, call the inner function to sort
    // this set of values, then send a message to the UI to update it on our
    // progress

    if ((end - begin) < pSortInfo->nProgressChunk)
    {
        __int64 iNow;
        __int64 iDuration;

        // Let the inner function do the work

        qsort_inner (pSortInfo, begin, end);

        // Get the current time, calculate how long we've taken and tell the UI
        // about our progress

        iNow = GetTime();
        iDuration = iNow - pSortInfo->iStart;

        ::PostMessage (pSortInfo->hWnd, WMU_SERIAL_BUMP_PROGRESS, end - begin, (LPARAM)iDuration);
        return;
    }

    // If we're not done, partition the data and recurse on the two subsets.

    if (begin != end)
    {
        --end;  // Exclude last element (pivot) from partition

        int *middle = partition (begin, end, bind2nd (less<typename iterator_traits<int *>::value_type>(), *begin));

        using std::swap;
        swap(*end, *middle);                        // Move pivot to middle

        qsort (pSortInfo, begin, middle);
        qsort (pSortInfo, ++middle, ++end);   // Exclude pivot and restore end
    }
}

//
// SerialQSort
//
// Thread procedure to generate pseudo-random data and sort it using the
// quicksort algorithm
//

unsigned __stdcall SerialQSort (void *p)
{
    __int64 iNow;
    __int64 iDuration;
    serial_qsort_info *pSortInfo = (serial_qsort_info *)p;

    // Let the UI know that we're alive and initializing

    ::PostMessage (pSortInfo->hWnd, WMU_SERIAL_INITIALIZING, 0, 0);

    // Allocate space for the data and generate random information to sort

    int *a = new int [pSortInfo->n];

    for (int i = 0; i < pSortInfo->n; ++i)
    {
        a[i] = 1000000001 * i;
    }

    // Get the current time and send a message to the UI that we've completed
    // our initialization and will start sorting the data

    pSortInfo->iStart = GetTime();
    ::PostMessage (pSortInfo->hWnd, WMU_SERIAL_STARTING, 0, 0);

    // Sort the data

    qsort (pSortInfo, a, a + pSortInfo->n);

    // Get the current time and calculate how long we took to sort the data.
    // Send how long we took to the UI.
    //
    // Note that we're ASSUMING that the high 32 bits of the duration are 0!
    //
    // Note that we can't let the UI calculate the times since we're using
    // PostMessage instead of SendMessage.  PostMessage puts the message on the
    // UI's input queue and doesn't wait for the UI to process it.  We must use
    // PostMessage because MFC maintains state in thread-local storage and this
    // isn't an MFC thread, and it isn't the thread that created the dialog box.

    iNow = GetTime();
    iDuration = iNow - pSortInfo->iStart;

    ::PostMessage (pSortInfo->hWnd, WMU_SERIAL_DONE, (WPARAM)iDuration, 0);

    // Clean up and exit the thread procedure, which will kill the thread cleanly.

    delete [] a;

    return 0;
}
