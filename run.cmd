@echo off
REM Run anything in the project Isaac Lab environment, from any shell or directory:
REM   run.cmd experiments\watch.py --mode passive
REM
REM The interpreter lives at the environment ROOT (standard standalone-install
REM layout: python.exe + python311.dll + DLLs\ + Lib\ side by side). Do NOT put
REM copies of python*.dll into Scripts\ -- they shadow the real ones and break
REM every compiled extension module (h5py, etc.) with ENTRYPOINT_NOT_FOUND.
"%~dp0env_isaaclab\python.exe" %*
