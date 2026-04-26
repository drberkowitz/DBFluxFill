@echo off
:: DBFluxFill Setup
:: Downloads embedded Python 3.11 for inference, finds system Python for the installer GUI.
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set INSTALLER=%SCRIPT_DIR%installer.py
set PYTHON_DIR=%SCRIPT_DIR%python
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PYTHON_ZIP_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
set PYTHON_ZIP=%SCRIPT_DIR%python-embed.zip

:: ---------------------------------------------------------------
:: PHASE 1 - Embedded Python for inference (skip if already done)
:: ---------------------------------------------------------------

if exist "%PYTHON_EXE%" (
    echo Embedded Python already set up. Skipping download.
    goto :FIND_SYSTEM_PYTHON
)

:: Check if zip was manually placed in the folder
if exist "%PYTHON_ZIP%" (
    echo Found python-embed.zip in folder. Using local copy.
    goto :UNPACK
)

echo Downloading embedded Python 3.11...
powershell -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_ZIP_URL%' -OutFile '%PYTHON_ZIP%' -UseBasicParsing -ErrorAction Stop } catch { Write-Host $_.Exception.Message; exit 1 }"
if !errorlevel! neq 0 (
    echo.
    echo ---------------------------------------------------------------
    echo  Download failed. This may be due to a firewall or no internet.
    echo ---------------------------------------------------------------
    echo.
    echo  To install manually:
    echo.
    echo  1. On a machine with internet access, download this file:
    echo     %PYTHON_ZIP_URL%
    echo.
    echo  2. Copy the downloaded zip into this folder:
    echo     %SCRIPT_DIR%
    echo.
    echo  3. Make sure it is named exactly:
    echo     python-embed.zip
    echo.
    echo  4. Re-run this setup.bat
    echo.
    pause
    exit /b 1
)

:UNPACK
echo Unpacking embedded Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
if !errorlevel! neq 0 (
    echo Failed to unpack embedded Python.
    pause
    exit /b 1
)

del "%PYTHON_ZIP%" >nul 2>&1

echo Patching python311._pth to enable site-packages...
powershell -Command "(Get-Content '%PYTHON_DIR%\python311._pth') -replace '#import site','import site' | Set-Content '%PYTHON_DIR%\python311._pth'"

echo Embedded Python ready.

:: ---------------------------------------------------------------
:: PHASE 2 - Find system Python for the installer GUI
:: ---------------------------------------------------------------

:FIND_SYSTEM_PYTHON
set GUI_PYTHON=

:: Try Windows Launcher first
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3 -c "import tkinter" >nul 2>&1
    if !errorlevel! equ 0 (
        set GUI_PYTHON=py -3
        goto :RUN_INSTALLER
    )
)

:: Try python on PATH
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import tkinter; import sys; exit(0 if sys.version_info >= (3,6) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set GUI_PYTHON=python
        goto :RUN_INSTALLER
    )
)

:: Try python3 on PATH
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    python3 -c "import tkinter; import sys; exit(0 if sys.version_info >= (3,6) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set GUI_PYTHON=python3
        goto :RUN_INSTALLER
    )
)

:: Nothing found - give the user clear instructions
echo.
echo ---------------------------------------------------------------
echo  Python 3.6 or later is needed to run the DBFluxFill installer.
echo ---------------------------------------------------------------
echo.
echo  It does not appear to be installed on this machine.
echo  Python 3.11 is recommended.
echo.
echo  Would you like to download and install Python 3.11 automatically?
echo.
choice /C YN /N /M "Install Python 3.11 automatically? [Y/N]: "
if !errorlevel! equ 2 goto :MANUAL_INSTRUCTIONS

:: User chose Y - download and install
echo.
echo Downloading Python 3.11 installer...
set PYTHON_INSTALLER=%SCRIPT_DIR%python311-installer.exe
set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

powershell -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_INSTALLER_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing -ErrorAction Stop } catch { Write-Host $_.Exception.Message; exit 1 }"
if !errorlevel! neq 0 (
    echo.
    echo Download failed. Please install Python 3.11 manually.
    goto :MANUAL_INSTRUCTIONS
)

echo Installing Python 3.11...
start /wait "" "%PYTHON_INSTALLER%" /quiet PrependPath=1 Include_launcher=1
del "%PYTHON_INSTALLER%" >nul 2>&1

:: Verify it worked
py -3 -c "import tkinter" >nul 2>&1
if !errorlevel! equ 0 (
    set GUI_PYTHON=py -3
    echo Python 3.11 installed successfully.
    goto :RUN_INSTALLER
)
python -c "import tkinter" >nul 2>&1
if !errorlevel! equ 0 (
    set GUI_PYTHON=python
    echo Python 3.11 installed successfully.
    goto :RUN_INSTALLER
)

echo.
echo Python 3.11 was installed but could not be detected yet.
echo This sometimes requires a restart. Please restart your
echo computer and re-run setup.bat.
echo.
pause
exit /b 1

:MANUAL_INSTRUCTIONS
echo.
echo ---------------------------------------------------------------
echo  How to install Python 3.11 manually
echo ---------------------------------------------------------------
echo.
echo  1. Open this link in your browser:
echo     https://www.python.org/downloads/release/python-3119/
echo.
echo  2. Scroll to the bottom of the page.
echo.
echo  3. In the table under Files, click:
echo     Windows installer ^(64-bit^)
echo.
echo  4. Run the downloaded installer.
echo.
echo  5. IMPORTANT: On the first screen of the installer,
echo     check the box that says:
echo     "Add Python 3.11 to PATH"
echo     before clicking Install Now.
echo.
echo  6. Once installation is complete, close this window
echo     and re-run setup.bat.
echo.
echo  If your studio network blocks python.org, ask your
echo  IT department to install Python 3.11 for you, making
echo  sure the "Add to PATH" option is selected.
echo.
pause
exit /b 1

:RUN_INSTALLER
echo.
echo Launching DBFluxFill Setup...
echo.
%GUI_PYTHON% "%INSTALLER%"
echo.
if %errorlevel% neq 0 (
    echo Installer exited with an error. See above for details.
    pause
)