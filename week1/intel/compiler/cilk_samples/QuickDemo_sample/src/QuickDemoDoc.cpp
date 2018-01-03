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

// QuickDemoDoc.cpp : implementation of the CQuickDemoDoc class
//


#include "stdafx.h"
#include "QuickDemo.h"

#include "QuickDemoDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CQuickDemoDoc

IMPLEMENT_DYNCREATE(CQuickDemoDoc, CDocument)

BEGIN_MESSAGE_MAP(CQuickDemoDoc, CDocument)
END_MESSAGE_MAP()


// CQuickDemoDoc construction/destruction

CQuickDemoDoc::CQuickDemoDoc()
{
	// TODO: add one-time construction code here

}

CQuickDemoDoc::~CQuickDemoDoc()
{
}

BOOL CQuickDemoDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CQuickDemoDoc serialization

void CQuickDemoDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CQuickDemoDoc diagnostics

#ifdef _DEBUG
void CQuickDemoDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CQuickDemoDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CQuickDemoDoc commands
