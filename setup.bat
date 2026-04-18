@echo off
:: DBFluxFill Setup
:: Launches installer.py using any Python 3.x
:: installer.py handles Python 3.11 detection and installation

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set INSTALLER=%SCRIPT_DIR%installer.py
set PYTHON_VERSION=3.11.9
set PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

:: Try Windows Python Launcher for any Python 3
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.6 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_EXE=py -3.6
        goto :RUN_INSTALLER
    )
)

:: Try python on PATH, check if it's 3.x
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import sys; v=sys.version_info; exit(0 if (v.major == 3 and v.minor >= 6) or v.major > 3 else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_EXE=python
        goto :RUN_INSTALLER
    )
)

:: Python 3.11 not found - ask user if they want to install it
echo.
echo Python 3.11 not found in your system.
echo DBFluxFill requires Python 3.11 to run.
echo.
echo Would you like Python 3.11 to be downloaded and installed for you?
echo.
choice /C YN /N /M "Install Python 3.11? [Y/N]: "
set RESPONSE=!errorlevel!
if !RESPONSE! equ 2 (
    echo ----------------------------------------------------------
    echo To install Python 3.11 manually:
    echo ----------------------------------------------------------
    echo 1. Open: https://www.python.org/downloads/release/python-3119/
    echo 2. At the bottom, select 'Windows installer ^(64-bit^)' in the table under 'Version'
    echo 3. Run the installer being sure to check 'Add Python to PATH'
    echo 4. Relaunch this setup bat file
    echo.
    pause
    exit /b 1
) else if !RESPONSE! equ 1 (
    echo Downloading Python 3.11 installer...
    set "PYTHON_INSTALLER=%SCRIPT_DIR%python-3.11-installer.exe"
    echo Attempting download from: %PYTHON_DOWNLOAD_URL%
    powershell -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_DOWNLOAD_URL%' -OutFile \"!PYTHON_INSTALLER!\" -UseBasicParsing -ErrorAction Stop } catch { exit 1 }"
    if !errorlevel! neq 0 (
        echo Failed to download Python 3.11 installer.
        echo Error code: !errorlevel!
        echo Please download it manually from: https://www.python.org/downloads/release/python-3119/
        echo.
        pause
        exit /b 1
    )
    echo Download completed successfully.
    if exist "!PYTHON_INSTALLER!" (
        echo Installing Python 3.11 ^(this may take a minute^)...
        start /wait "" "!PYTHON_INSTALLER!" /quiet PrependPath=1
        del "!PYTHON_INSTALLER!" >nul 2>&1
        for /f "tokens=2*" %%i in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Python\PythonCore\3.11\InstallPath" /ve 2^>nul') do (
            set "PYTHON_EXE=%%jpython.exe"
        )
        if defined PYTHON_EXE (
            if exist "!PYTHON_EXE!" (
                echo Python 3.11 installed successfully!
                echo.
                goto :RUN_INSTALLER
            )
        )
        echo.
        echo Python 3.11 installation may have failed or the registry entry was not created.
        echo Please restart your terminal or computer, then run this setup again.
        echo.
        pause
        exit /b 1
    ) else (
        echo Failed to download Python 3.11 installer.
        echo Please download it manually from: https://www.python.org/downloads/release/python-3119/
        echo.
        pause
        exit /b 1
    )
) else (
    echo ERROR: Unexpected response value: !RESPONSE!
    echo Expected: 1 for Yes, 2 for No
    pause
    exit /b 1
)

:RUN_INSTALLER
:: Run the installer
echo Launching DBFluxFill Setup...
echo.
%PYTHON_EXE% "%INSTALLER%"