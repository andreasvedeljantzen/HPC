
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
// pi_console.cpp : Defines the entry point for the console application.


#include "stdafx.h"
#include <iostream>
#include "pi.h"
#define NUM_STEPS 100000000 //default number of steps used to calculate Pi

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	if( argc >2)
	{
		cout << "Usage: pi.exe <number> where <number> x 100000000 is the number of steps in Pi computation" << endl;
		cout << "<number> between 1 and 20 is accepted, Default is 1" <<endl;
		cout << "Type options: -h|? for help"<<endl;
		return 1;
	}
	long nSteps = 0L;
	if( argc == 2)
	{
		if( argv[1][0] == '?' || (argv[1][0] == '-' && (argv[1][1] == 'h')))
		{
			cout << "Usage: pi.exe <number> where <number> x 100000000 is the number of steps in Pi computation" << endl;
			cout << "<number> between 1 and 20 is accepted, Default is 1" <<endl;
			return 0;
		}
		nSteps = atol(argv[1]);
		if( (nSteps >= 1) && (nSteps <= 20))
		{
			nSteps *= NUM_STEPS;
		}else{
			cout << "Please provide a number between 1 and 20" << nSteps << endl;
			return 0;
		}
	} 

	if( argc == 1)
	{
		nSteps = NUM_STEPS;	

	}
	piGetSolutions ( nSteps);
	return 0;
}

