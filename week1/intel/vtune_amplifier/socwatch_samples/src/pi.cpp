//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2015 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ===============================================================
// [DESCRIPTION]
// Calculate the value of pi over the given number of steps.
// 
// [RUN]
// pi_console.exe <num> * 1 million by default where num is integer between 1 and 20

#include "stdafx.h"
#include <mmsystem.h>
#include <iostream>

using namespace std;
CString GetStampString()
{
	SYSTEMTIME curTime;
    GetLocalTime(&curTime);

	CString strOutText = CString(""); 
	strOutText.AppendFormat("%02d/%02d, %02d:%02d:%02d", curTime.wMonth, curTime.wDay, curTime.wHour, curTime.wMinute, curTime.wSecond);

	return strOutText; 
}


double CalcSum(long i,double sum, double step)
{
	double x;
	x = (i + .5)*step;
	sum = sum + 4.0/(1.+ x*x);
	return sum;
}

double CalcPi(long num_steps)
{
	long i;
	double pi;
	double sum=0.0;
	double step;

	step = 1./ num_steps;

	for (i=0; i<num_steps; i++)
	{
		sum = CalcSum(i, sum, step);
	}
	
	pi = sum*step;

 	return pi;
}

void piGetSolutions(long nSteps)
{
	CString strStamp = GetStampString();
	CString strOutText("Starting pi solver for steps "); 
	strOutText.AppendFormat("%ld, starting %s\r\n", nSteps, strStamp);

	double pi;

	DWORD startTime=timeGetTime();

	// calc the solutions. Could have the user input the number of steps, 
	// but the timing in seconds only gets interesting after 100,000,000 steps.
	// So, we used the #define NUM_STEPS instead.
	pi = CalcPi(nSteps);

	DWORD dwTimeUsed = timeGetTime()-startTime;

	strOutText.AppendFormat("    Value of Pi: %f\r\n    Calculations took: %u ms.\r\n\r\n", pi, dwTimeUsed);
	cout << strOutText << endl;
}
