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
// QSort_Parallel.cpp
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// This file implements a parallel quicksort using Intel(R) Cilk(TM) Plus.
//

#include "stdafx.h"
#include <algorithm>
#include <iterator>
#include <functional>
#include <math.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <windows.h>

#include "QSort_Parallel.h"
#include "GetTime.h"
#include "UserMessages.h"

#pragma warning(disable: 1684)  // Suppress complaint about pointer to same-sized integral type

using namespace std;

//
// qsort_inner
// 
// Simple parallel implementation of QSort.  The inner function doesn't have
// to pass around information for the messages to the UI.
//

static
void qsort_inner (parallel_qsort_info *pSortInfo, int *begin, int *end)
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
// Simple parallel implementation of QSort.  If we get below nProgressChunk,
// switch to the inner function and post a message to the dialog when that
// chunk of work is done.
//

static
void qsort (parallel_qsort_info *pSortInfo, int *begin, int *end)
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

        ::PostMessage (pSortInfo->hWnd, WMU_PARALLEL_BUMP_PROGRESS, end - begin, (LPARAM)iDuration);

        // Tell the UI about the work we just completed so it can be displayed

        int iWorker = __cilkrts_get_worker_number();

        ::PostMessage (pSortInfo->hWnd, WMU_PARALLEL_COLOR_PROGRESS,
                       (iWorker << 24) | (end - begin),
                       (LPARAM)begin);    
        return;
    }

    // If we're not done, partition the data and recurse on the two subsets.
    // We only spawn the first subset since there's no benefit from letting
    // anyone steal the small bit of code before the implied sync at the end
    // of the function

    if (begin != end)
    {
        --end;  // Exclude last element (pivot) from partition

        int *middle = partition (begin, end, bind2nd (less<typename iterator_traits<int *>::value_type>(), *begin));

        using std::swap;
        swap(*end, *middle);                        // Move pivot to middle

        cilk_spawn qsort (pSortInfo, begin, middle);
        qsort (pSortInfo, ++middle, ++end);   // Exclude pivot and restore end
        cilk_sync;
    }
}

//
// do_cilk_qsort
//
// Allocate the data, do the sort, and keep the UI informed of what we're doing
//

static
int do_cilk_qsort (parallel_qsort_info *pSortInfo)
{
    __int64 iNow;
    __int64 iDuration;

    // Allocate space for the data and generate random information to sort

    int *a = new int [pSortInfo->n];

    for (int i = 0; i < pSortInfo->n; ++i)
    {
        a[i] = 1000000001 * i;
    }

    // Get the current time and send a message to the UI that we've completed
    // our initialization and will start sorting the data

    pSortInfo->iStart = GetTime();
    ::PostMessage (pSortInfo->hWnd, WMU_PARALLEL_STARTING, (WPARAM)a, 0);

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
    // PostMessage because MFC maintains state in thread-local storage and
    // Intel(R) Cilk(TM) Plus' scheduling among it's worker threads would confuse it
    // (to put it mildly).

    iNow = GetTime();
    iDuration = iNow - pSortInfo->iStart;

    ::PostMessage (pSortInfo->hWnd, WMU_PARALLEL_DONE, (WPARAM)iDuration, 0);

    // Clean up and return

    delete [] a;

    return 0;
}

//
// ParallelQSort
//
// Thread procedure to generate pseudo-random data and sort it using the
// Quick Sort algorithm
//

unsigned __stdcall ParallelQSort (void *p)
{
    parallel_qsort_info *pSortInfo = (parallel_qsort_info *)p;

    // Let the UI know that we're alive and initializing

    ::PostMessage (pSortInfo->hWnd, WMU_PARALLEL_INITIALIZING, 0, 0);

    // Restrict the number of threads the Intel(R) Cilk(TM) runtime will use.  By default
    // Intel(R) Cilk(TM) Plus will create a worker thread for each processor in the
    // system.  But we (probably) want to reserve one for the serial sort

    if (pSortInfo->iWorkers)
    {
        char buf[16];
        sprintf_s(buf, 16, "%d", pSortInfo->iWorkers);
        __cilkrts_set_param("nworkers", buf);
    }

    // Start the Intel(R) Cilk(TM) Plus quick sort implementation

    do_cilk_qsort (pSortInfo);

    // Shut down the Intel(R) Cilk(TM) Plus runtime so the next run can change the
    // number of workers

    __cilkrts_end_cilk();

    // Exit the thread procedure, which will kill the thread cleanly.

    return 0;
}
