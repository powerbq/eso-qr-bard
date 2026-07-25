@echo off

if not exist build mkdir build

cd build

pyinstaller --clean --distpath dist --workpath work ..\app.spec

pause
