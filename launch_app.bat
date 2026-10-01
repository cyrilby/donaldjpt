@echo off
call .venv\Scripts\activate
python "src/donaldjpt/app.py"
call deactivate