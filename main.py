import csv
from datetime import datetime
import os
import threading
import numpy as np
import winsound
import cv2
from ultralytics import YOLO

# 1. Load Pre-trained YOLOv8 Nano Model
model = YOLO('yolov8n.pt')

# GPU/CPU Warm-up to eliminate the first-frame inference freeze
dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
_ = model(dummy_frame, verbose=False)

# 2. Source Configuration
VIDEO_SOURCE = os.path.join('assets', 'ambulance.mp4')
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
  print(f"Error: Unable to open video source '{VIDEO_SOURCE}'.")
  exit()

# 3. CSV Audit Logger Initialization
LOG_FILE = 'corridor_log.csv'
with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
  writer = csv.writer(f)
  if f.tell() == 0:
    writer.writerow(
        ['Timestamp', 'Detected_Vehicle', 'Signal_Status', 'Action_Taken']
    )


def log_emergency_event_async(vehicle_type, status, action):
  """Writes log asynchronously to prevent file I/O latency."""

  def _write():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      writer.writerow([timestamp, vehicle_type, status, action])

  threading.Thread(target=_write, daemon=True).start()


def play_beep_async():
  """Plays alert beep on a daemon thread."""
  try:
    winsound.Beep(1200, 300)
  except Exception:
    pass


# State Control Variables
green_corridor_active = False
has_logged = False
frame_count = 0
STARTUP_DELAY_FRAMES = 20  # Keeps RED signal for the first ~0.7 seconds

print('[INFO] Smart Green-Corridor AI System Active... Monitoring feed.')

while cap.isOpened():
  success, frame = cap.read()

  if not success:
    print('[INFO] Video stream ended safely.')
    break

  frame_count += 1
  frame = cv2.resize(frame, (960, 540))

  detected_box = None
  vehicle_label = ''

  # Continuous inference running with warm model
  results = model(frame, conf=0.55, verbose=False)

  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0])
      name = model.names[cls_id]

      if name in ['truck', 'car', 'bus']:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        if width > 140 and height > 140:
          detected_box = (x1, y1, x2, y2)
          vehicle_label = name
          break
    if detected_box:
      break

  # Trigger green corridor only after initial visual delay
  if detected_box and frame_count > STARTUP_DELAY_FRAMES:
    x1, y1, x2, y2 = detected_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
    cv2.putText(
        frame,
        'EMERGENCY AMBULANCE DETECTED',
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
    )

    green_corridor_active = True

    if not has_logged:
      has_logged = True
      log_emergency_event_async(
          f'Ambulance ({vehicle_label})',
          'GREEN',
          'Priority Green Corridor Active',
      )
      threading.Thread(target=play_beep_async, daemon=True).start()

  # HUD Rendering
  if green_corridor_active:
    cv2.rectangle(frame, (0, 0), (960, 55), (0, 180, 0), -1)
    cv2.putText(
        frame,
        '*** EMERGENCY CORRIDOR: GREEN SIGNAL GRANTED ***',
        (120, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.circle(frame, (910, 28), 16, (0, 255, 0), -1)
  else:
    cv2.rectangle(frame, (0, 0), (960, 50), (30, 30, 30), -1)
    cv2.putText(
        frame,
        'TRAFFIC STATUS: NORMAL | SIGNAL: RED',
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.circle(frame, (910, 25), 15, (0, 0, 255), -1)

  cv2.imshow('Smart Green-Corridor AI System', frame)

  if cv2.waitKey(33) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()