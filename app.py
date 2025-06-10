ImportError: Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.
Traceback:
File "C:\Users\2179048\AppData\Local\Programs\Python\Python311\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 600, in _run_script
    exec(code, module.__dict__)
File "C:\Users\2179048\Desktop\visualization\app.py", line 14, in <module>
    df = pd.read_excel(uploaded_file)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\2179048\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\io\excel\_base.py", line 495, in read_excel
    io = ExcelFile(
         ^^^^^^^^^^
File "C:\Users\2179048\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\io\excel\_base.py", line 1567, in __init__
    self._reader = self._engines[engine](
                   ^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\2179048\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\io\excel\_openpyxl.py", line 552, in __init__
    import_optional_dependency("openpyxl")
File "C:\Users\2179048\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\compat\_optional.py", line 138, in import_optional_dependency
    raise ImportError(msg)
