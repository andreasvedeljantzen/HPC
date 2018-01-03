@echo off

set OUTPUTDIR=bin
set MY_INCLUDE=src\
SET PROJECT_NAME=socwatch_samples
set EXEC=Pi_Console.exe
set PROJDIR=msvc\
set DESTDIR=bin\Debug\
set BINDIR=bin\

if /i "%1"=="clean" goto clean

:options
if "%1"=="" goto compile
shift
goto options

:compile
mkdir %DESTDIR% 2>nul
devenv %PROJDIR%%PROJECT_NAME%.vcxproj /build
goto eof

:clean
devenv %PROJDIR%%PROJECT_NAME%.vcxproj /clean
echo removing files...
:eof
