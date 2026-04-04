# FlowSync

A decentralized traffic intelligence system that uses AI to detect vehicle density at intersections and automatically control traffic lights in real time — no centralized server, no constant internet required.

---

## How It Works

**AI Detection**
YOLOv8 Nano counts vehicles in each lane from video or image input. Lanes are ranked by density — highest to lowest.

**Decision System**
The system assigns green light time based on vehicle count. Highest density lane goes first, then medium, then lowest. After all three lanes complete, it rescans and repeats.

**API + ESP32**
Flask serves a REST API at `/traffic`. The ESP32 Dev Board polls it every second and controls the physical traffic lights. A live browser dashboard runs at `/`.

---

## How to Run

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Add video or image files**

Drop `.mp4` or `.jpg` files into the `test_media/` folder.

**Start the system**
```bash
python traffic_ai_video.py
```

**Open the dashboard**
```
http://127.0.0.1:5000
```

---

*Made in Nigeria.*
