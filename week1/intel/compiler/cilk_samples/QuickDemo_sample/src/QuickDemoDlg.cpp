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
// QuickDemoDlg.cpp
// This file is part of the QuickSort demonstration of Intel(R) Cilk(TM) Plus.
//
// This file implements CQuickDemoDlg, a CDialog-derived class.  CQuickDemoDlg
// provides the UI for the QuickDemo application.
//

#include "stdafx.h"
#include "AboutBox.h"
#include <process.h>
#include <time.h>
#include "QuickDemo.h"
#include "QuickDemoDlg.h"
#include "QSort_Serial.h"
#include "GetTime.h"
#include "UserMessages.h"
#include <windows.h>

#include <cilk/cilk.h>

#pragma warning(disable: 1684)  // Suppress complaint about pointer to same-sized integral type

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define WORK_UNIT_DIVISOR 100
#define PROGRESS_BAR_DIVISIONS 100

// CQuickDemoDlg dialog

CQuickDemoDlg::CQuickDemoDlg(CWnd* pParent /*=NULL*/)
: CDialog(CQuickDemoDlg::IDD, pParent)
{
    m_lSortCount = 0;
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
    m_nUnits = 10 * 1000 * 1000;	// 10 Million
    m_nProgressBarUnits = m_nUnits / PROGRESS_BAR_DIVISIONS;
    m_iSerialPauseTime = 0;
    m_iParallelPauseTime = 0;
    m_bSerialDone = false;
    m_bParallelDone = false;
}

void CQuickDemoDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_SERIAL_STATIC, m_ctlStaticSerialStatus);
    DDX_Control(pDX, IDC_SERIAL_PROGRESS, m_ctlProgressSerial);
    DDX_Control(pDX, IDC_PARALLEL_PROGRESS, m_ctlProgressParallel);
    DDX_Control(pDX, IDC_PARALLEL_STATIC, m_ctlStaticParallelStatus);
    DDX_Control(pDX, IDC_THREADS_COMBO, m_ctlWorkersCombo);
    DDX_Control(pDX, IDC_PARALLEL_PROGRESS_STATIC, m_ctlStaticParallelProgress);
    DDX_Control(pDX, IDC_PAUSE_BUTTON, m_ctlPauseButton);
    DDX_Control(pDX, IDC_SERIAL_BUTTON, m_ctlSerialButton);
    DDX_Control(pDX, IDC_PARALLEL_BUTTON, m_ctlParallelButton);
    DDX_Control(pDX, IDC_START_BOTH_BUTTON, m_ctlBothSortsButton);
    DDX_Control(pDX, IDC_PROPORTION_STATIC, m_ctlProportionStatic);
    DDX_Control(pDX, IDC_WORKERS_STATIC, m_ctlThreadsStatic);
}

BEGIN_MESSAGE_MAP(CQuickDemoDlg, CDialog)
    ON_WM_SYSCOMMAND()
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    ON_MESSAGE (WMU_SERIAL_INITIALIZING, OnSerialInitializing)
    ON_MESSAGE (WMU_SERIAL_STARTING, OnSerialStarting)
    ON_MESSAGE (WMU_SERIAL_BUMP_PROGRESS, OnSerialBumpProgress)
    ON_MESSAGE (WMU_SERIAL_DONE, OnSerialDone)
    ON_MESSAGE (WMU_PARALLEL_INITIALIZING, OnParallelInitializing)
    ON_MESSAGE (WMU_PARALLEL_STARTING, OnParallelStarting)
    ON_MESSAGE (WMU_PARALLEL_BUMP_PROGRESS, OnParallelBumpProgress)
    ON_MESSAGE (WMU_PARALLEL_DONE, OnParallelDone)
    ON_MESSAGE (WMU_PARALLEL_COLOR_PROGRESS, OnParallelColorProgress)
    //}}AFX_MSG_MAP
    ON_BN_CLICKED(IDC_PAUSE_BUTTON, &CQuickDemoDlg::OnBnClickedPauseButton)
    ON_BN_CLICKED(IDC_SERIAL_BUTTON, &CQuickDemoDlg::OnBnClickedSerialButton)
    ON_BN_CLICKED(IDC_PARALLEL_BUTTON, &CQuickDemoDlg::OnBnClickedParallelButton)
    ON_BN_CLICKED(IDC_START_BOTH_BUTTON, &CQuickDemoDlg::OnBnClickedStartBothButton)
END_MESSAGE_MAP()


// CQuickDemoDlg message handlers

void CQuickDemoDlg::SetWindowTitle()
{
    CString num, oldnum, str;
    unsigned value, remainder;
    const wchar_t *pszDivider = L"";
    const wchar_t *pszFormat;

    value = m_nUnits;
    while (value > 0)
    {
        remainder = value % 1000;
        value = value / 1000;

        if (value > 0)
            pszFormat = L"%03d%s%s";
        else
            pszFormat = L"%d%s%s";

        num.Format (pszFormat, remainder, pszDivider, oldnum);
        pszDivider = L",";
        oldnum = num;
    }

    str.Format (L"QuickDemo - Quicksort of %s integers", (const wchar_t *)num);
    SetWindowText (str);
}

BOOL CQuickDemoDlg::OnInitDialog()
{
    CDialog::OnInitDialog();

    // Add "About..." menu item to system menu.

    // IDM_ABOUTBOX must be in the system command range.
    ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
    ASSERT(IDM_ABOUTBOX < 0xF000);

    CMenu* pSysMenu = GetSystemMenu(FALSE);
    if (pSysMenu != NULL)
    {
        CString strAboutMenu;
        strAboutMenu.LoadString(IDS_ABOUTBOX);
        if (!strAboutMenu.IsEmpty())
        {
            pSysMenu->AppendMenu(MF_SEPARATOR);
            pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
        }
    }

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);			// Set big icon
    SetIcon(m_hIcon, FALSE);		// Set small icon

    // Set the window title with the number of integers we'll be sorting

    SetWindowTitle ();

    // Determine how many CPUs there are

    SYSTEM_INFO info;
    ::GetSystemInfo(&info);

    // Initialize the thread combobox.  Default to one less than the number of
    // cores so they can run the serial version on 1 and the Intel(R) Cilk(TM) Plus
    // version on the others.  For now we're limiting it to 16 threads because
    // there's only 16 colors defined in CProgressStatic & beyond that there's
    // probably too many slices in the display

    int item = -1;
    int sel_item = -1;
    CString str;
    unsigned nDefaultSelectedCpus = max (1, info.dwNumberOfProcessors - 1);

    for (unsigned i = 1; i <= 16; i++)
    {
        str.Format (L"%d", i);
        item = m_ctlWorkersCombo.AddString (str);
        if (nDefaultSelectedCpus == i)
            sel_item = item;
        m_ctlWorkersCombo.SetItemData (item, i);
    }

    m_ctlWorkersCombo.SetCurSel (sel_item);

    // Create the event that can be used to pause the sort

    m_hPauseEvent = CreateEvent (NULL, TRUE, TRUE, NULL);

    return TRUE;  // return TRUE  unless you set the focus to a control
}

void CQuickDemoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
    if ((nID & 0xFFF0) == IDM_ABOUTBOX)
    {
        CAboutDlg dlgAbout;
        dlgAbout.DoModal();
    }
    else
    {
        CDialog::OnSysCommand(nID, lParam);
    }
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CQuickDemoDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CQuickDemoDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

//
// CommonProgress
//
// Common code for both the parallel and serial sort to update the UI with
// progress sorting the data
//

LRESULT CQuickDemoDlg::CommonProgress (CProgressCtrl &ctlProgress,
                                       CStatic &ctlStatus,
                                       unsigned nUnitsCompleted,
                                       LPARAM lDuration,
                                       __int64 iPauseTime)
{
    // Update the progress bar with how far we've gotten

    int nPos = nUnitsCompleted / m_nProgressBarUnits;
    ctlProgress.SetPos (nPos);

    // Convert how long we've taken into a readable format.  lDuration is the
    // low 32bits of a FILETIME.  It's *ASSUMED* that the high 32bits are 0...

    SYSTEMTIME st;
    FILETIME tDuration;
    tDuration.dwHighDateTime = 0;
    tDuration.dwLowDateTime = (DWORD)(lDuration - iPauseTime);

    FileTimeToSystemTime (&tDuration, &st);
    CString csStatus;
    csStatus.Format (L"Running: %d.%03d (seconds)", (60 * st.wMinute) + st.wSecond, st.wMilliseconds);

    ctlStatus.SetWindowTextW (csStatus);

    return 0;		// We've handled the message
}

//
// CommonDone
//
// Common code for both the parallel and serial sort to update the UI with
// the final run time
//

LRESULT CQuickDemoDlg::CommonDone (CProgressCtrl &ctlProgress,
                                   CStatic &ctlStatus,
                                   WPARAM wDuration,
                                   __int64 iPauseTime)
{
    // Force the progress bar to 100% done

    ctlProgress.SetPos (PROGRESS_BAR_DIVISIONS);

    // Convert the duration into a readable format and display it

    SYSTEMTIME st;
    FILETIME tDuration;
    tDuration.dwHighDateTime = 0;
    tDuration.dwLowDateTime = (DWORD)(wDuration - iPauseTime);

    FileTimeToSystemTime (&tDuration, &st);
    CString csStatus;
    csStatus.Format (L"Finished: %d.%03d (seconds)", (60 * st.wMinute) + st.wSecond, st.wMilliseconds);

    ctlStatus.SetWindowTextW (csStatus);

    // Disable the pause button.  The button won't actually gray until the sort
    // count goes to zero

    EndASort();

    return 0;		// We've handled the message
}

//
// OnSerialInitializing
//
// Message from the serial sort that it's generating random data to sort
//
// wParam - Not used
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnSerialInitializing (WPARAM wParam, LPARAM lParam)
{
    m_ctlStaticSerialStatus.SetWindowTextW (L"Initializing...");
    m_ctlProgressSerial.SetRange (0, PROGRESS_BAR_DIVISIONS);
    m_nSerialUnitsCompleted = 0;

    return 0;		// We've handled the message
}

//
// OnSerialStarting
//
// Message from the serial sort that it's completed initialization and
// started to sort the data
//
// wParam - Not used
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnSerialStarting (WPARAM wParam, LPARAM lParam)
{
    m_ctlStaticSerialStatus.SetWindowTextW (L"Running...");

    return 0;		// We've handled the message
}

//
// OnSerialBumpProgress
//
// Message from the serial sort on progress it's made.
//
// wParam - # of units it's completed working on
// lParam - Low 32 bits of the FILETIME for how long it's taken to get to this
//			point.  It's *ASSUMED* that the high 32 bits are zero...
//

afx_msg LRESULT CQuickDemoDlg::OnSerialBumpProgress (WPARAM wParam, LPARAM lParam)
{
    m_nSerialUnitsCompleted += (unsigned)wParam;
    return CommonProgress (m_ctlProgressSerial,
                           m_ctlStaticSerialStatus,
                           m_nSerialUnitsCompleted,
                           lParam,
                           m_iSerialPauseTime);
}

//
// OnSerialDone
//
// Message from the serial sort that it's done.
//
// wParam - Low 32 bits of the FILETIME for the sort's duration.  It's
//			*ASSUMED* that the high 32 bits are zero...
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnSerialDone (WPARAM wParam, LPARAM lParam)
{
    m_bSerialDone = true;
    m_dwSerialDuration = (DWORD)wParam - (DWORD)m_iSerialPauseTime;

    return CommonDone (m_ctlProgressSerial,
                       m_ctlStaticSerialStatus,
                       wParam,
                       m_iSerialPauseTime);
}

//
// OnBnClickedSerialButton()
//
// Called when the user clicks the button to start the serial sort.  Spin off
// a thread to run it.
//

void CQuickDemoDlg::OnBnClickedSerialButton()
{
    unsigned nThreadId;
    uintptr_t hThread;

    // Update the UI - enabling/disabling buttons to reflect that the sort
    // is running

    StartASort();

    // Gather the information for the serial quicksort

    m_SerialInfo.iStart = 0;
    m_SerialInfo.hWnd = this->m_hWnd;
    m_SerialInfo.hPause = m_hPauseEvent;
    m_SerialInfo.lPause = false;
    m_SerialInfo.n = m_nUnits;
    m_SerialInfo.nProgressChunk = m_nUnits / WORK_UNIT_DIVISOR;

    // Spin off a thread to run the serial quicksort

    hThread = _beginthreadex (NULL,			// Security
        0,			// Stack size
        SerialQSort,	// Thread function
        &m_SerialInfo,// Thread parameter
        0,			// Init flag
        &nThreadId);	// Recieves thread ID
}

//
// OnParallelInitializing
//
// Message from the parallel sort that it's generating random data to sort
//
// wParam - Not used
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnParallelInitializing (WPARAM wParam, LPARAM lParam)
{
    m_ctlStaticParallelStatus.SetWindowTextW (L"Initializing...");
    m_ctlProgressParallel.SetRange (0, PROGRESS_BAR_DIVISIONS);
    m_nParallelUnitsCompleted = 0;
    m_bSerialDone = m_bParallelDone = false;

    return 0;		// We've handled the message
}

//
// OnParallelStarting
//
// Message from the parallel sort that it's completed initialization and
// started to sort the data
//
// wParam - Base address of the data - we'll want it for the colored bar
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnParallelStarting (WPARAM wParam, LPARAM lParam)
{
    m_ctlStaticParallelStatus.SetWindowTextW (L"Running...");

    m_pParallelData = (int *)wParam;

    return 0;		// We've handled the message
}

//
// OnParallelBumpProgress
//
// Message from the parallel sort on progress it's made.
//
// wParam - # of units it's completed working on
// lParam - Low 32 bits of the FILETIME for how long it's taken to get to this
//			point.  It's *ASSUMED* that the high 32 bits are zero...
//

afx_msg LRESULT CQuickDemoDlg::OnParallelBumpProgress (WPARAM wParam, LPARAM lParam)
{
    m_nParallelUnitsCompleted += (unsigned)wParam;
    return CommonProgress (m_ctlProgressParallel,
                           m_ctlStaticParallelStatus,
                           m_nParallelUnitsCompleted,
                           lParam,
                           m_iParallelPauseTime);
}

//
// OnParallelDone
//
// Message from the parallel sort that it's done.
//
// wParam - Low 32 bits of the FILETIME for the sort's duration.  It's
//			*ASSUMED* that the high 32 bits are zero...
// lParam - Not used
//

afx_msg LRESULT CQuickDemoDlg::OnParallelDone (WPARAM wParam, LPARAM lParam)
{
    m_bParallelDone = true;
    m_dwParallelDuration = (DWORD)wParam - (DWORD)m_iParallelPauseTime;

    // Call the common code to update the UI with the final run time

    return CommonDone (m_ctlProgressParallel,
                       m_ctlStaticParallelStatus,
                       wParam,
                       m_iParallelPauseTime);
}

//
// OnParallelColorProgress
//
// Message from the parallel sort that it's done.
//
// wParam - WorkerID and number of units completed
// lParam - Address in the data array for start of completed work
//

afx_msg LRESULT CQuickDemoDlg::OnParallelColorProgress (WPARAM wParam, LPARAM lParam)
{
    int iWorker = (int)(wParam >> 24);			// High 8 bits are the worker ID
    int iUnits = (int)(0x00ffffff & wParam);	// Low 24 bits are number of units
    int *begin = (int *)lParam;

    m_ctlStaticParallelProgress.ShowWork (iWorker,
        (int)(begin - m_pParallelData),
        iUnits,
        m_nUnits);
    return 0;
}

//
// OnBnClickedParallelButton
//
// The user clicked the "Start Parallel Sort" button, so spin off a thread to
// run it.
//

void CQuickDemoDlg::OnBnClickedParallelButton()
{
    unsigned nThreadId;
    uintptr_t hThread;

    // Update the UI - enabling/disabling buttons to reflect that the sort
    // is running

    StartASort();

    // Gather the information for the parallel quicksort

    m_ParallelInfo.iStart = 0;
    m_ParallelInfo.hWnd = this->m_hWnd;
    m_ParallelInfo.hPause = m_hPauseEvent;
    m_ParallelInfo.lPause = false;
    m_ParallelInfo.n = m_nUnits;
    m_ParallelInfo.nProgressChunk = m_nUnits / WORK_UNIT_DIVISOR;

    m_ParallelInfo.iWorkers = 1;
    int item = m_ctlWorkersCombo.GetCurSel();
    if (-1 != item)
        m_ParallelInfo.iWorkers = (int)m_ctlWorkersCombo.GetItemData (item);

    m_ctlStaticParallelProgress.SetWorkerCount (m_ParallelInfo.iWorkers);

    // Spawn a thread to run the parallel quicksort

    hThread = _beginthreadex (NULL,				// Security
                              0,				// Stack size
                              ParallelQSort,	// Thread function
                              &m_ParallelInfo,	// Thread parameter
                              0,				// Init flag
                              &nThreadId);		// Recieves thread ID
}

//
// OnBnClickedStartBothButton
//
// The user clicked the "Start Both Sorts" button, so start them both
//

void CQuickDemoDlg::OnBnClickedStartBothButton()
{
    OnBnClickedParallelButton();
    OnBnClickedSerialButton();
}

//
// OnBnClickedPauseButton
//
// The user clicked the "Pause" button which toggles between "Pause Sort" and
// "Resume Sort".  Depending on our current state, either pause the running
// sort(s), or let them continue running.
//

void CQuickDemoDlg::OnBnClickedPauseButton()
{
    // Grab the current time

    __int64 iNow = GetTime();

    // What we do depends on whether we're currently paused

    if (m_bPaused)
    {
        // We're currently paused.  Start by adding the time we spent paused to
        // m_iPauseTime, which will be subtracted from the time(s) for the sort(s)
        // since the user doesn't want the elapsed sort time to include the time
        // spent paused

        if (! m_bParallelDone)
            m_iParallelPauseTime += iNow - m_iPauseStart;
        if (! m_bSerialDone)
            m_iSerialPauseTime += iNow - m_iPauseStart;

        // Change the button label to reflect what it will do next time

        m_ctlPauseButton.SetWindowText (L"Pause Sort");

        // Toggle the saved state & allow the sort(s) to proceed.  We use both
        // global variables and an event to avoid constant kernel transitions
        // to check the event.

        m_bPaused = false;
        m_ParallelInfo.lPause = false;
        m_SerialInfo.lPause = false;
        ::SetEvent (m_hPauseEvent);
    }
    else
    {
        // We're currently running.  Reset the event & flags which will cause the
        // sort(s) to pause, and toggle our state.  We use both global variables
        // and an event to avoid constant kernel transitions to check the event.

        ::ResetEvent (m_hPauseEvent);
        m_SerialInfo.lPause = true;
        m_ParallelInfo.lPause = true;
        m_bPaused = true;

        // Change the window text so the user knows how to resume

        m_ctlPauseButton.SetWindowText (L"Resume Sort");

        // Save the current time so we can figure out how long we were paused

        m_iPauseStart = iNow;
    }
}

//
// StartASort
//
// Update the UI to reflect that a sort is running.  This function can be
// called multiple times (once per parallel/serial sort).  The first time
// in, it will reset the Paused flag and matching event, as well as enable
// the "Pause" button and disable the "Start Sort" buttons.
//

void CQuickDemoDlg::StartASort()
{
    // Make sure we only do this once per run

    long lSortCount;
    lSortCount = ::InterlockedIncrement (&m_lSortCount);
    if (1 != lSortCount)
        return;

    // Reset the progress bars

    m_ctlProgressSerial.SetPos (0);
    m_ctlProgressParallel.SetPos (0);

    // Initialize the paralell progress static.  It will be filled with colored
    // bands indicating which worker worked on each "chunk"

    m_ctlStaticParallelProgress.Reset();

    // Initialize the sort durations

    m_dwSerialDuration = 0;
    m_dwParallelDuration = 0;

    // Reset the "Paused" toggle state, and set the event so no threads wait
    // on it accidentally

    m_bPaused = false;
    ::SetEvent (m_hPauseEvent);

    // Make sure the pause times are reset

    m_iSerialPauseTime = 0;
    m_iParallelPauseTime = 0;

    // Make sure the "Pause" is enabled and has the correct text

    m_ctlPauseButton.EnableWindow ();
    m_ctlPauseButton.SetWindowText (L"Pause Sort");

    // Disable the sort start buttons & thread combo

    m_ctlSerialButton.EnableWindow (FALSE);
    m_ctlParallelButton.EnableWindow (FALSE);
    m_ctlBothSortsButton.EnableWindow (FALSE);
    m_ctlWorkersCombo.EnableWindow (FALSE);
    m_ctlThreadsStatic.EnableWindow (FALSE);

    // No results yet, so we can't show the relative performance

    m_ctlProportionStatic.SetWindowText (L"");
    m_ctlStaticParallelStatus.SetWindowText (L"");
    m_ctlStaticSerialStatus.SetWindowText (L"");
}

//
// EndASort
//
// Update the UI to reflect that a sort has finished.  This function can be
// called multiple times (once per parallel/serial sort).  When the count of
// running sorts reaches 0, the "Pause" button will be disabled and the
// "Start Sort" buttons will be enabled.
//

void CQuickDemoDlg::EndASort()
{
    // If this all sorts aren't finished, keep waiting

    long lSortCount;
    lSortCount = ::InterlockedDecrement (&m_lSortCount);
    if (0 != lSortCount)
        return;

    // Disable the "Pause" button and make sure it has the right text

    m_ctlPauseButton.EnableWindow (FALSE);
    m_ctlPauseButton.SetWindowText (L"Pause Sort");

    // Reenable the sort start buttons & threads combo

    m_ctlSerialButton.EnableWindow (TRUE);
    m_ctlParallelButton.EnableWindow (TRUE);
    m_ctlBothSortsButton.EnableWindow (TRUE);
    m_ctlWorkersCombo.EnableWindow (TRUE);
    m_ctlThreadsStatic.EnableWindow (TRUE);

    if ((0 != m_dwSerialDuration) && (m_dwParallelDuration != 0))
    {
        CString str;
        if (m_dwParallelDuration < m_dwSerialDuration)
            str.Format (L"Speed-up: Serial sort time was %.2f x Parallel sort time",
            (double)m_dwSerialDuration / (double)m_dwParallelDuration);
        else
            str.Format (L"Slow-down: Parallel sort time was %.2f x Serial sort time",
            (double)m_dwParallelDuration / (double)m_dwSerialDuration);
        m_ctlProportionStatic.SetWindowText (str);
    }
}
