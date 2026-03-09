# Vision Memory Assistant — Installation Guide

A personal item re-identification system that helps you remember where you last saw your belongings using your phone's camera.

---

## What You'll Need

- **Windows PC** (Windows 10 or later)
- **Python 3.10** installed ([Download from python.org](https://www.python.org/downloads/))
- **Android phone** with IP Webcam app ([Download from Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam))
- **Same WiFi network** for both your PC and phone

---

## Step-by-Step Installation

### 1. Clone the Project from GitHub

Open **Command Prompt** (search for "cmd" in Windows Start menu) and run:

```bash
git clone https://github.com/DONYsins/personal-object-memory-assistant.git
```

Navigate to the project folder:

```bash
cd personal-object-memory-assistant
```

### 2. Install Required Packages

Install all required packages:

```bash
pip install -r requirements.txt
```
*Note: On a machine with a compatible NVIDIA GPU, PyTorch with CUDA (GPU support) can be installed. **Make sure that the python version is 3.10**. Detection processes faster with GPU*

*On a machine without a Nvidia GPU, PyTorch will still install, but it will run in CPU mode and may show a warning about missing CUDA.*

*This may take 5-10 minutes depending on your internet speed.*

```bash
# Run this if your Python version is compatible and GPU is available
# PyTorch with CUDA 12.1 (Install the version that supports your device)
torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Set Up Your Phone Camera

1. Install **IP Webcam** app on your Android phone from Play Store
2. Open the app and scroll to the bottom
3. Tap **"Start Server"**
4. Note the URL shown (e.g., `http://192.168.1.100:8080/video`)

**Important:** Make sure your phone and PC are connected to the same WiFi network!

---

## Running the Application

You need to run **two separate Command Prompt windows** — one for the backend server and one for the web interface.

### Terminal 1 — Start Backend Server

Open a Command Prompt window and run:

```bash
cd personal-object-memory-assistant
uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
```

**Keep this window open** — you should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Start Web Interface

Open a **second** Command Prompt window and run:

```bash
cd personal-object-memory-assistant
streamlit run ui_streamlit.py --server.port 8501 --server.address localhost
```

**Keep this window open too** — your browser will automatically open to `http://localhost:8501`

---

## First Time Setup

1. **Register an account** — Create your username and password
2. **IP Camera URL** - Enter the camera stream URL (e.g. http://192.168.1.100:8080/video) into the IP Camera URL field
2. **Add your objects** — Tell the system what items you want to track (e.g., "Blue Watch", "car_keys")
3. **Enroll your objects** — Point your phone camera at each item for 1-2 minutes from different angles
4. **Define your rooms** — Create environments like "Bedroom", "Living Room"
5. **Add landmarks** — Define furniture in each room (e.g., "white Chair", "master_bed")
6. **Enroll landmarks** — Point your camera at each piece of furniture for 1-2 minutes in different angles
7. **Test Environemnt** — Runs the video and shows the object detection of the added objects and landmarks in live
8. **Start tracking** — Let the system watch where you place your items!

---

## How to Use

### Tracking Mode
1. Go to **"Tracking"** section
2. Enter your phone's camera URL (from IP Webcam app)
3. Click **"Start Tracking"**
4. Place items in view of your camera
5. The system automatically records where items are seen
6. Click **"Stop Tracking"** once you are done

### Query Mode
1. Go to **"Query"** section
2. Type what you're looking for (e.g., "where is my blue watch?")
3. View the last 10 locations with photos and timestamps

---

## Troubleshooting

### "Camera failed to start"
- Check that IP Webcam app is running on your phone
- Verify the camera URL is correct
- Make sure PC and phone are on the same WiFi

### "Module not found" errors
- Run `pip install -r requirements.txt` again
- Make sure you're in the correct project folder

### Application won't start
- Close all Command Prompt windows
- Delete the `logs/.current_session` file (if it exists)
- Try running both terminal commands again

### Slow detection
- Use a phone with good lighting
- Keep camera stable
- Reduce camera resolution in IP Webcam settings (Settings → Video Resolution → 640×480)

---

## Need More Help?


---

**Enjoy tracking your belongings! 🎯**