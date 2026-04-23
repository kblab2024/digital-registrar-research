===========================================================
  Digital Registrar - Pathology Report Annotator (Dummy)
===========================================================

This folder is self-contained. No Python installation needed.


使用方式 / Usage
----------------
1. 將此資料夾整份解壓縮到任意位置 (例如桌面)。
   Extract this folder to any location (e.g. Desktop).

2. 雙擊 `run.bat`。
   Double-click `run.bat`.

   * 第一次執行時 Windows 可能會詢問是否允許防火牆 -
     請選擇「允許存取」(Allow access)。
     On first launch Windows may ask about the firewall -
     choose "Allow access".

3. 瀏覽器會自動開啟 http://localhost:8501。
   The browser opens http://localhost:8501 automatically.

4. 於側邊欄選擇 annotator 後開始標註。
   Select an annotator in the sidebar, then start annotating.

5. 關閉 `run.bat` 視窗以停止伺服器
   (按 Ctrl+C 或直接關閉該視窗)。
   Close the `run.bat` window to stop the server.


資料位置 / Data
---------------
預標註與報告 / Pre-annotations and reports:
  dummy\data\<dataset>\reports\
  dummy\data\<dataset>\preannotation\<model>\...

儲存的標註 / Saved annotations:
  dummy\data\<dataset>\annotations\<annotator>_<mode>\...

您的標註會直接寫入到上述資料夾，可隨時備份。
Your saved annotations write directly into the folders above;
back them up whenever you like.


疑難排解 / Troubleshooting
-------------------------
* 瀏覽器沒有自動開啟:
  手動前往 http://localhost:8501
  Browser did not open automatically:
  manually navigate to http://localhost:8501.

* 8501 連接埠被其他程式佔用:
  編輯 `run.bat`, 將 `--server.port=8501` 改成其他數字
  (例如 8502), 再重新執行。
  Port 8501 is already in use:
  edit `run.bat`, change `--server.port=8501` to e.g. 8502,
  then run again.

* `run.bat` 視窗一閃而過看不到錯誤:
  打開「命令提示字元」(Command Prompt), 把 run.bat 拖進去
  執行, 即可看到錯誤訊息。
  The `run.bat` window flashes and closes without showing
  an error: open Command Prompt and drag `run.bat` into it
  to keep the window open.


系統需求 / Requirements
----------------------
* Windows 10 / 11 (64-bit)
* ~200 MB disk space (Python and all dependencies are bundled)
* No Python installation needed.
