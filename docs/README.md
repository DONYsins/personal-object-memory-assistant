# Documentation Index — Vision Memory Assistant

A personal item re-identification system using YOLOv8, CLIP, and FAISS to track
where personal objects (Watch, Wallet, Keys) were last seen, using an IP camera
and room-level environment context.

---

## Documents

| # | File | What's inside |
|---|---|---|
| 0 | [00_README_QuickStart.md](00_README_QuickStart.md) | Setup, run commands, happy-path demo, common errors and fixes |
| 1 | [01_Architecture_Overview.md](01_Architecture_Overview.md) | Component diagram, all 5 sequence diagrams, why 3 processes, what a session is (plain English) |
| 2 | [02_Backend_API.md](02_Backend_API.md) | Every endpoint: method, params, sample request/response, DB reads/writes, files created |
| 3 | [03_UI_Flow_and_Buttons.md](03_UI_Flow_and_Buttons.md) | Every page, every button: what it calls, what state changes, what the user sees |
| 4 | [04_Vision_Pipeline.md](04_Vision_Pipeline.md) | Frame format, YOLO output parsing, CLIP embedding (with math), FAISS search, location inference with examples |
| 5 | [05_Data_Storage_Model.md](05_Data_Storage_Model.md) | Full DB schema, file/folder layout, logging strategy, data retention and cleanup |
| 6 | [06_Design_Choices_and_Justifications.md](06_Design_Choices_and_Justifications.md) | Why each library was chosen, what alternatives were rejected, architectural decisions justified |

---

## Key File Map (code ↔ docs)

| Source file | Primary doc |
|---|---|
| `backend_api.py` | [02](02_Backend_API.md), [04](04_Vision_Pipeline.md), [05](05_Data_Storage_Model.md) |
| `constants.py` | [00](00_README_QuickStart.md) §7, [06](06_Design_Choices_and_Justifications.md) |
| `ingest_ipcam.py` | [01](01_Architecture_Overview.md) §3b–d, [04](04_Vision_Pipeline.md) §7 |
| `logger.py` | [05](05_Data_Storage_Model.md) §3 |
| `ui_streamlit.py` | [03](03_UI_Flow_and_Buttons.md) |
| `ui_components/dashboard.py` | [03](03_UI_Flow_and_Buttons.md) |
| `ui_components/login.py` | [03](03_UI_Flow_and_Buttons.md), [01](01_Architecture_Overview.md) §3a |
| `ui_components/query.py` | [03](03_UI_Flow_and_Buttons.md) |
| `ui_components/tracking.py` | [03](03_UI_Flow_and_Buttons.md), [01](01_Architecture_Overview.md) §3d |
| `ui_components/environment.py` | [03](03_UI_Flow_and_Buttons.md) |
| `ui_components/utils.py` | [03](03_UI_Flow_and_Buttons.md) |
