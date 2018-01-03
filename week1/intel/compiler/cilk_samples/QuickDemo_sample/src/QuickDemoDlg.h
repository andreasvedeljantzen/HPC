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
// QuickDemoDlg.h
// This file declares CQuickDemoDlg, a CDialog-derived class.  CQuickDemoDlg
// provides the UI for the QuickDemo application.
//

#pragma once
#include "afxwin.h"
#include "afxcmn.h"
#include "QSort_Parallel.h"
#include "QSort_Serial.h"
#include "ProgressStatic.h"

// CQuickDemoDlg dialog
class CQuickDemoDlg : public CDialog
{
// Construction
public:
        CQuickDemoDlg(CWnd* pParent = NULL);	// standard constructor


// Dialog Data
        enum { IDD = IDD_QUICKDEMO_DIALOG };

        protected:
        virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

        void SetWindowTitle();

        afx_msg LRESULT OnSerialInitializing (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnSerialStarting (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnSerialBumpProgress (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnSerialDone (WPARAM wParam, LPARAM lParam);

        afx_msg LRESULT OnParallelInitializing (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnParallelStarting (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnParallelBumpProgress (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnParallelDone (WPARAM wParam, LPARAM lParam);
        afx_msg LRESULT OnParallelColorProgress (WPARAM wParam, LPARAM lParam);

        LRESULT CommonDone (CProgressCtrl &ctlProgress,
                                                CStatic &ctlStatus,
                                                WPARAM wDuration,
                                                __int64 iPauseTime);
        LRESULT CommonProgress (CProgressCtrl &ctlProgress,
                                CStatic &ctlStatus,
                                unsigned nUnitsCompleted,
                                LPARAM lDuration,
                                __int64 iPauseTime);

// Implementation
        // Generated message map functions
        virtual BOOL OnInitDialog();
        afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
        afx_msg void OnPaint();
        afx_msg HCURSOR OnQueryDragIcon();
        DECLARE_MESSAGE_MAP()

        void StartASort();
        void EndASort();

public:
        afx_msg void OnBnClickedSerialButton();
        afx_msg void OnBnClickedParallelButton();
        afx_msg void OnBnClickedStartBothButton();
        afx_msg void OnBnClickedPauseButton();

private:
        CComboBox m_ctlWorkersCombo;
        CProgressCtrl m_ctlProgressParallel;
        CStatic m_ctlStaticParallelStatus;
        CProgressCtrl m_ctlProgressSerial;
        CStatic m_ctlStaticSerialStatus;
        CStatic m_ctlProportionStatic;
        CProgressStatic m_ctlStaticParallelProgress;
        CButton m_ctlPauseButton;
        CButton m_ctlSerialButton;
        CButton m_ctlParallelButton;
        CButton m_ctlBothSortsButton;
        CBitmap m_bitmapStaticParallelProgress;
        CStatic m_ctlThreadsStatic;
        CRect m_rectStaticParallelProgress;
        HICON m_hIcon;

        unsigned m_nUnits;
        unsigned m_nProgressBarUnits;
        unsigned m_nSerialUnitsCompleted;
        unsigned m_nParallelUnitsCompleted;
        DWORD m_dwSerialDuration;
        DWORD m_dwParallelDuration;
        int *m_pParallelData;
        HANDLE m_hPauseEvent;
        volatile long m_lSortCount;
        bool m_bPaused;
        bool m_bSerialDone;
        bool m_bParallelDone;
        __int64	m_iPauseStart;
        __int64 m_iSerialPauseTime;
        __int64 m_iParallelPauseTime;
        serial_qsort_info m_SerialInfo;
        int m_nUnused[32];					// Force parallel & serial info to diff. cache lines
        parallel_qsort_info m_ParallelInfo;
public:
};
