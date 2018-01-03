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
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// Declaration of the CProgressStatic control which displays which thread has completed
// which portion of the work
//

#pragma once

// CProgressStatic

class CProgressStatic : public CStatic
{
	DECLARE_DYNAMIC(CProgressStatic)

public:
	CProgressStatic();
	virtual ~CProgressStatic();

	// Implementation

	void SetWorkerCount (int nWorkers);
	void ShowWork (int iWorker, int iStart, int iRun, int iTotalSpan);
	void Reset (bool bInvalidate=true);

	// Overrides

	virtual void PreSubclassWindow();
	virtual void DrawItem (LPDRAWITEMSTRUCT lpDrawItemStruct);

protected:
    afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	DECLARE_MESSAGE_MAP()

private:
	bool		m_bInitialized;
	int			m_nWorkers;
	COLORREF   *m_clrWorkers;
	CRect		m_rectClient;
	CBitmap		m_bitmap;
};
