===========================================================
  Digital Registrar - Compare / Consensus Tool
===========================================================

This folder is self-contained. No Python installation needed.

Purpose:
  Side-by-side comparison of two annotators' work (NHC and KPC,
  with_preann mode) and editing of the consolidated "gold" annotation
  that downstream evaluation scripts consume.


使用方式 / Usage
----------------
1. 將此資料夾整份解壓縮到任意位置 (例如桌面)。
   Extract this folder anywhere (e.g. Desktop).

2. 雙擊 `run.bat` 啟動正式比對工具。
   Double-click `run.bat` to launch the production tool.
     - 資料根目錄 / data root: `workspace\`
     - 標註者鎖定 / annotators locked: NHC vs KPC
     - 模式鎖定 / mode locked:        Consensus, with_preann
     - 輸出 / output:                workspace\data\<dataset>\
                                       annotations\gold\<n>\<case_id>.json

3. (選用 / optional) 雙擊 `run_demo.bat` 啟動展示模式。
   Optional: double-click `run_demo.bat` to explore the full UI
   against the empty `dummy\` skeleton (no real data).
     - 模式選單可見 / mode picker visible
     - 預設指向 `dummy\` (空骨架，僅供示範介面)

4. 第一次執行時 Windows 可能會詢問是否允許防火牆 -
   請選擇「允許存取」(Allow access)。
   On first launch Windows may prompt about the firewall — choose
   "Allow access".

5. 瀏覽器會自動開啟 http://localhost:8501。
   The browser opens http://localhost:8501 automatically.

6. 關閉啟動視窗以停止伺服器 (按 Ctrl+C 或直接關閉該視窗)。
   Close the launcher window to stop the server (Ctrl+C or just
   close the window).


Workflow / 工作流程
-------------------
On the left of the screen, A (NHC) and B (KPC) annotations are
shown side-by-side per field, with disagreements highlighted. The
Gold column is editable. Each row has "⇐ Use A" / "⇐ Use B"
shortcuts to copy that side's value into Gold. Multi-value array
fields also offer a "∪ Union" shortcut.

Press the "🏁 Save Gold" button to write the consolidated gold
annotation. The file is saved to:

   workspace\data\<dataset>\annotations\gold\<n>\<case_id>.json

This is the canonical layout consumed by the eval pipeline (rule /
ClinicalBERT / LLM baselines, ablations, etc.). No further
post-processing is needed.


資料放置 / Place your dataset
-----------------------------
將您的資料依下列結構放入 `workspace\` 資料夾：
Place your data inside `workspace\` using this structure:

   workspace\data\<dataset>\
       reports\<organ_n>\<case_id>.txt
       annotations\nhc_with_preann\<organ_n>\<case_id>.json
       annotations\kpc_with_preann\<organ_n>\<case_id>.json
       annotations\gold\<organ_n>\<case_id>.json   ← (written by this tool)

可參考同層 `dummy\` 資料夾的範例骨架。
See the sibling `dummy\` folder for the directory skeleton.

A case appears in the sample list only when the report file
(`reports\<n>\<case_id>.txt`) exists. The two annotator JSON files
are read if present; the Gold panel is pre-filled from A on first
view, then loaded back from disk if a Gold file already exists for
the case.


疑難排解 / Troubleshooting
-------------------------
* 瀏覽器沒有自動開啟:
  手動前往 http://localhost:8501
  Browser did not open automatically: navigate manually.

* 8501 連接埠被其他程式佔用:
  編輯 run.bat, 將 `--server.port=8501` 改為其他數字。
  Port 8501 in use: edit run.bat to change the port.

* `run.bat` 視窗一閃而過看不到錯誤:
  打開「命令提示字元」(Command Prompt), 把 run.bat 拖進去執行。
  Window flashes shut: open Command Prompt and drag run.bat into it
  to keep the window open.

* "尚未載入樣本" 訊息:
  確認 `workspace\data\<dataset>\reports\<n>\` 下有 .txt 檔案。
  "No samples loaded": confirm reports/ has .txt files.


系統需求 / Requirements
----------------------
* Windows 10 / 11 (64-bit)
* ~250 MB disk space (Python and all dependencies are bundled)
* No Python installation needed.
