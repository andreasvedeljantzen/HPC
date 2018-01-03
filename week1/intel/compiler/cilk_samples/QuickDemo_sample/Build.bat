@echo off
SETLOCAL

rem CommandPromptType is set in both VS and Intel, but not in normal DOS environment
if /i "%VCINSTALLDIR%"=="" (
	echo "environment not set up (needs Intel environment)"
	goto eof
)
set OUTFNAME=QuickDemo
set EXEC=%OUTFNAME%.exe
set SAMPLENAME=Quick Demo
set SRCDIR=src\
rem default is release build
set DESTDIR=release\

rem PRODUCT_NAME is only defined in Intel environment
rem if in Visual Studio environment
if /i "%PRODUCT_NAME%"=="" (
	echo "environment not set up (needs Intel environment)"
	goto eof
)

rem else if in Intel environment
if /i NOT "%PRODUCT_NAME%"=="" (
	set CC=icl
	set LINKER=xilink
)

set LINK_FLAGS=/INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFEST:NO


if /i "%1"=="clean" goto clean
if /i "%1"=="run" goto run
if /i "%1"=="help" goto help
if /i "%1"=="debug" goto debug
if /i "%1"=="release" goto release

rem default is release build
goto release

:debug
set DESTDIR=debug\
set RCFLAGS=/D "DEBUG" /D "_AFXDLL" /D "_UNICODE" /D "UNICODE" /l 0x409 
set CFLAGS=/c /Od /EHsc /MDd /GS /W3 /Zi /Qdiag-disable:1885
set PCHFLAG=/Yc"StdAfx.h" /Fp"%DESTDIR%QuickDemo.pch" 
set PCHFLAG_USE=/Yu"StdAfx.h" /Fp"%DESTDIR%QuickDemo.pch"  
set DefFlags=/D "WIN32" /D "_WINDOWS" /D "DEBUG" /D "_AFXDLL" /D "_UNICODE" /D "UNICODE" 

goto compile_and_link

:release
set DESTDIR=release\
set RCFLAGS=/D "NDEBUG" /D "_AFXDLL" /D "_UNICODE" /D "UNICODE" /l 0x409 
set CFLAGS=/c /O2 /Qipo /EHsc /MD /GS /fp:fast /W3 /Zi /Qdiag-disable:1885
set PCHFLAG=/Yc"StdAfx.h" /Fp"%DESTDIR%QuickDemo.pch" 
set PCHFLAG_USE=/Yu"StdAfx.h" /Fp"%DESTDIR%QuickDemo.pch"  
set DefFlags=/D "WIN32" /D "_WINDOWS" /D "NDEBUG" /D "_AFXDLL" /D "_UNICODE" /D "UNICODE" 


:compile_and_link
mkdir %DESTDIR% 2>nul
echo on

rc.exe %RCFLAGS% /I %DESTDIR% /fo"%DESTDIR%QuickDemo.res" %srcDir%QuickDemo.rc

icl %CFLAGS% %DefFlags% /Fd%DESTDIR% /Fo%DESTDIR% %PCHFLAG% %srcDir%stdafx.cpp

icl %CFLAGS% %DefFlags% /Fd%DESTDIR% /Fo%DESTDIR% %PCHFLAG_USE% %srcDir%AboutBox.cpp %srcDir%ChildFrm.cpp %srcDir%GetTime.cpp %srcDir%MainFrm.cpp %srcDir%ProgressStatic.cpp %srcDir%QSort_Parallel.cpp %srcDir%QSort_Serial.cpp %srcDir%QuickDemo.cpp %srcDir%QuickDemoDlg.cpp %srcDir%QuickDemoDoc.cpp %srcDir%QuickDemoView.cpp 

xilink.exe /OUT:"%DESTDIR%%OUTFNAME%.exe" /INCREMENTAL:NO /MANIFEST /MANIFESTFILE:"%DESTDIR%%OUTFNAME%.exe.intermediate.manifest" /TLBID:1 /DEBUG /PDB:"%DESTDIR%%OUTFNAME%.pdb" /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /ENTRY:"wWinMainCRTStartup" /IMPLIB:"%DESTDIR%%OUTFNAME%.lib" /FIXED:NO %DESTDIR%stdafx.obj %DESTDIR%AboutBox.obj %DESTDIR%ChildFrm.obj %DESTDIR%GetTime.obj %DESTDIR%MainFrm.obj %DESTDIR%ProgressStatic.obj %DESTDIR%QSort_Parallel.obj %DESTDIR%QSort_Serial.obj %DESTDIR%QuickDemo.obj %DESTDIR%QuickDemoDlg.obj %DESTDIR%QuickDemoDoc.obj %DESTDIR%QuickDemoView.obj %DESTDIR%QuickDemo.res

mt.exe /outputresource:"%DESTDIR%%OUTFNAME%.exe;#1" /manifest %DESTDIR%%OUTFNAME%.exe.intermediate.manifest

@echo off
goto eof

:run
%DESTDIR%%EXEC% %2
goto eof

:help
echo "Syntax: build [debug|release|run|clean]"
echo "     build debug - Build %SAMPLENAME% without optimization"
echo "     build release - Build %SAMPLENAME% with optimization"
echo "     build run - Run %SAMPLENAME%"
echo "     build clean - Clean build directory"
goto eof

:clean
echo removing files...
rmdir /Q /S %DESTDIR% 2>nul

:eof
ENDLOCAL
