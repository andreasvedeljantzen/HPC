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

// QuickDemoView.cpp : implementation of the CQuickDemoView class
//


#include "stdafx.h"
#include "QuickDemo.h"

#include "QuickDemoDoc.h"
#include "QuickDemoView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CQuickDemoView

IMPLEMENT_DYNCREATE(CQuickDemoView, CView)

BEGIN_MESSAGE_MAP(CQuickDemoView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
END_MESSAGE_MAP()

// CQuickDemoView construction/destruction

CQuickDemoView::CQuickDemoView()
{
	// TODO: add construction code here

}

CQuickDemoView::~CQuickDemoView()
{
}

BOOL CQuickDemoView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CQuickDemoView drawing

void CQuickDemoView::OnDraw(CDC* /*pDC*/)
{
	CQuickDemoDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
}


// CQuickDemoView printing

BOOL CQuickDemoView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CQuickDemoView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CQuickDemoView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CQuickDemoView diagnostics

#ifdef _DEBUG
void CQuickDemoView::AssertValid() const
{
	CView::AssertValid();
}

void CQuickDemoView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CQuickDemoDoc* CQuickDemoView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CQuickDemoDoc)));
	return (CQuickDemoDoc*)m_pDocument;
}
#endif //_DEBUG


// CQuickDemoView message handlers
