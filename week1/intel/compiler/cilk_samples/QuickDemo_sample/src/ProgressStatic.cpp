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
// ProgressStatic.h
//
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// Implementation of the CProgressStatic control which displays which thread
// has completed which portion of the work
//

#include "stdafx.h"
#include "QuickDemo.h"
#include "ProgressStatic.h"

// Default color for each thread

COLORREF clrDefaults[] =
{
	RGB (255,   0,   0),	// Red
	RGB (  0, 255,   0),	// Green
	RGB (  0,   0, 255),	// Blue
	RGB (255, 255,   0),	// Yellow
	RGB (255,   0, 255),	// Magenta
	RGB (  0, 255, 255),	// Cyan
	RGB (  0,   0,   0),	// Black
	RGB (255, 255, 255),	// White
	RGB (127,   0,   0),	// Dark Red
	RGB (  0, 127,   0),	// Dark Green
	RGB (  0,   0, 127),	// Dark Blue
	RGB (127, 127,   0),	// Dark Yellow
	RGB (127,   0, 127),	// Dark Magenta
	RGB (  0, 127, 127),	// Dark Cyan
	RGB ( 63,  63,  63),	// Dark Gray
	RGB (127, 127, 127),	// Gray
};

#define ARY_SIZE(_a) (sizeof(_a)/sizeof(_a[0]))

// CProgressStatic

IMPLEMENT_DYNAMIC(CProgressStatic, CStatic)

//
// CProgressStatic
//
// Construct an instance of the CProgressStatic control
//

CProgressStatic::CProgressStatic()
{
	m_nWorkers = 0;
	m_clrWorkers = NULL;
	m_bInitialized = false;
}

//
// ~CProgressStatic
//
// Destruct an instance of the CProgressStatic control
//

CProgressStatic::~CProgressStatic()
{
	if (m_clrWorkers)
		delete[] m_clrWorkers;
}


BEGIN_MESSAGE_MAP(CProgressStatic, CStatic)
    ON_WM_ERASEBKGND()
    ON_WM_DRAWITEM_REFLECT()
END_MESSAGE_MAP()

//
// PreSubclassWindow
//
// Ensure that the base static control has the SS_OWNERDRAW style set
//

void CProgressStatic::PreSubclassWindow()
{
	// Let the base class do it's thing

	CStatic::PreSubclassWindow();

	// Ensure that the SS_OWNERDRAW flag is set

	ModifyStyle (0, SS_OWNERDRAW);
}

//
// SetWorkerCount
//
// Set the number of workers
//

void CProgressStatic::SetWorkerCount(int nWorkers)
{
	// Release any previously allocated memory

	if (m_clrWorkers)
		delete[] m_clrWorkers;

	// Save the number of workers & allocate space for the color settings
	
	m_nWorkers = nWorkers;
	m_clrWorkers = new COLORREF[nWorkers];

	// Set the default colors.  Perhaps someday I'll allow someone to override
	// my defaults.

	int i, iColor = 0;
	for (i = 0; i < nWorkers; i++)
	{
		m_clrWorkers[i] = clrDefaults[iColor++];
		if (iColor >= ARY_SIZE(clrDefaults))
			iColor = 0;
	}
}

//
// OnEraseBkgnd
//
// Copy the bitmap onto the given DC.  I could do it here or in the DrawItem
// callback.  Do it here since they give us a convenient CDC *
//

BOOL CProgressStatic::OnEraseBkgnd(CDC* pDC)
{
	CBitmap *pOldBitmap;

	// Create a DC for the bitmap & save the current bitmap so we can restore
	// it later

	CDC dcMem;
	dcMem.CreateCompatibleDC (pDC);

	pOldBitmap = dcMem.GetCurrentBitmap();

	// If this is the first time in, get the control's client rectangle and
	// create a bitmap to hold what we're going to draw

	if (! m_bInitialized)
	{
		// Get the size of the static's client area

		GetClientRect(m_rectClient);

		// Create a compatible bitmap that's the size of the client area

		m_bitmap.CreateCompatibleBitmap (pDC,
										 m_rectClient.Width(),
										 m_rectClient.Height());

		// Fill the bitmap with the background color

		dcMem.SelectObject (m_bitmap);
		dcMem.FillSolidRect (m_rectClient, ::GetSysColor(COLOR_3DFACE)); // (this is meant for dialogs

		m_bInitialized = true;
	}

	// Select the bitmap into the DC we've created

	dcMem.SelectObject (m_bitmap);

	// Blt the bitmap onto the screen

	pDC->StretchBlt (0, 0,
					 m_rectClient.Width(), m_rectClient.Height(),
					 &dcMem,
					 0, 0,
					 m_rectClient.Width(), m_rectClient.Height(),
					 SRCCOPY);

	// Select our bitmap out of the created DC so it can be deleted

	dcMem.SelectObject (pOldBitmap);

    return TRUE;
}

//
// DrawItem
//
// Doesn't actually do anything, since we've done it all in EraseBkgnd
//

void CProgressStatic::DrawItem (LPDRAWITEMSTRUCT lpDIS)
{
}

//
// ShowWork
//
// Given a report from a thread that it's done some work, color a section of
// the control
//

void CProgressStatic::ShowWork (int iWorker, int iStart, int iRun, int iTotalSpan)
{
	CBitmap *pOldBitmap;

	// Create a DC for the bitmap

	CDC *pDC = GetDC();
	CDC dcMem;
	dcMem.CreateCompatibleDC (pDC);
	ReleaseDC (pDC);
	pDC = NULL;
	int x, cx;
	double dStep;

	// Calculate the starting position on the control and width we're supposed to
	// color.

	dStep = (double)m_rectClient.Width() / iTotalSpan;
	x = (int)(iStart * dStep);
	cx = (int)(iRun * dStep);
	cx++;

	// Select the bitmap into the DC, then color the portion corresponding to
	// the completed work

	pOldBitmap = dcMem.GetCurrentBitmap();
	dcMem.SelectObject (m_bitmap);
	dcMem.FillSolidRect (x, 0,
						 cx, m_rectClient.Height(),
						 m_clrWorkers[iWorker]);
	dcMem.SelectObject (pOldBitmap);

	// Invalidate the client area

	Invalidate();
}

//
// Reset
//
// Reset the client area to the background color
//

void CProgressStatic::Reset (bool bInvalidate)
{
	CBitmap *pOldBitmap;
	CDC *pDC = GetDC();
	CDC dcMem;
	dcMem.CreateCompatibleDC (pDC);
	ReleaseDC (pDC);
	pDC = NULL;

	pOldBitmap = dcMem.GetCurrentBitmap();
	dcMem.SelectObject (m_bitmap);
	dcMem.FillSolidRect (m_rectClient, ::GetSysColor(COLOR_3DFACE)); // (this is meant for dialogs
	dcMem.SelectObject (pOldBitmap);

	if (bInvalidate)
		Invalidate();
}
