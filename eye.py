import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from collections import deque
import time
import win32api




# Скачиваем модель
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("📥 Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded")

# MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("MediaPipe imports successful")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)
print("MediaPipe initialized")

# Индексы
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE_INDICES = [33, 133, 158, 159, 160, 173, 163, 145, 154]
RIGHT_EYE_INDICES = [362, 263, 386, 385, 398, 388, 381, 374, 396]

# Размер экрана пользователя
screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)
print(f"Screen size: {screen_width}x{screen_height}")

# Размеры окон (подстраиваем под разрешение)
window_width = screen_width // 2  # Половина ширины экрана
window_height = screen_height // 2  # Половина высоты экрана

print(f"Window size: {window_width}x{window_height}")

# ========== 32 ТОЧКИ КАЛИБРОВКИ ==========
calib_positions = []
margin = 50
cols = 8
rows = 4

step_x = (screen_width - 2 * margin) // (cols - 1)
step_y = (screen_height - 2 * margin) // (rows - 1)

for row in range(rows):
    for col in range(cols):
        x = margin + col * step_x
        y = margin + row * step_y
        calib_positions.append((x, y))

print(f"Created {len(calib_positions)} calibration points")


# ========== ФИЛЬТРЫ ==========
class LowPassFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.x = None
        self.y = None

    def filter(self, x, y):
        if self.x is None:
            self.x = x
            self.y = y
            return (x, y)

        self.x = int(self.alpha * x + (1 - self.alpha) * self.x)
        self.y = int(self.alpha * y + (1 - self.alpha) * self.y)
        return (self.x, self.y)


median_x = deque(maxlen=5)
median_y = deque(maxlen=5)
avg_x = deque(maxlen=7)
avg_y = deque(maxlen=7)
last_x = None
last_y = None
max_speed = 30
lowpass = LowPassFilter(alpha=0.25)

# Данные калибровки
calibration_points = []
calibration_targets = []
x_map = []
y_map = []
calibration_mode = False
current_calib_idx = 0

# Текущее положение взгляда
current_gaze = (screen_width // 2, screen_height // 2)


def apply_filters(x, y):
    #Применяет фильтры
    global last_x, last_y

    if last_x is not None:
        dx = x - last_x
        dy = y - last_y
        if abs(dx) > max_speed:
            x = last_x + (max_speed if dx > 0 else -max_speed)
        if abs(dy) > max_speed:
            y = last_y + (max_speed if dy > 0 else -max_speed)

    last_x, last_y = x, y

    median_x.append(x)
    median_y.append(y)

    if len(median_x) >= 3:
        x = int(np.median(median_x))
        y = int(np.median(median_y))

    x, y = lowpass.filter(x, y)

    avg_x.append(x)
    avg_y.append(y)

    if len(avg_x) >= 3:
        return (int(np.mean(avg_x)), int(np.mean(avg_y)))
    return (x, y)


def get_eye_center_and_size(face_landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    points = []
    for idx in eye_indices:
        landmark = face_landmarks[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    center_x = int(np.mean([p[0] for p in points]))
    center_y = int(np.mean([p[1] for p in points]))
    eye_width = max([p[0] for p in points]) - min([p[0] for p in points])
    eye_height = max([p[1] for p in points]) - min([p[1] for p in points])

    return (center_x, center_y), (eye_width, eye_height)


def get_relative_pupil(pupil, eye_center, eye_size):
    if eye_size[0] == 0 or eye_size[1] == 0:
        return None

    dx = pupil[0] - eye_center[0]
    dy = pupil[1] - eye_center[1]

    rel_x = dx / (eye_size[0] / 2)
    rel_y = dy / (eye_size[1] / 2)

    return (rel_x, rel_y)


def rel_to_screen(rel_x, rel_y):
    if len(x_map) < 5:
        return (int(screen_width // 2 + rel_x * 400),
                int(screen_height // 2 + rel_y * 300))

    x_dists = []
    for r, s in x_map:
        dist = abs(rel_x - r)
        x_dists.append((dist, s))

    x_dists.sort(key=lambda x: x[0])
    best_x = x_dists[:3]

    y_dists = []
    for r, s in y_map:
        dist = abs(rel_y - r)
        y_dists.append((dist, s))

    y_dists.sort(key=lambda x: x[0])
    best_y = y_dists[:3]

    screen_x = 0
    weight_sum = 0
    for dist, val in best_x:
        weight = 1.0 / (dist + 0.01)
        screen_x += val * weight
        weight_sum += weight
    screen_x = int(screen_x / weight_sum) if weight_sum > 0 else screen_width // 2

    screen_y = 0
    weight_sum = 0
    for dist, val in best_y:
        weight = 1.0 / (dist + 0.01)
        screen_y += val * weight
        weight_sum += weight
    screen_y = int(screen_y / weight_sum) if weight_sum > 0 else screen_height // 2

    return (screen_x, screen_y)


print("\nInitialization complete. Starting camera...")

# Запуск камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera!")
    exit()

# Создаем окно для камеры (подстраиваем под разрешение)
cam_width = window_width
cam_height = window_height
cv2.namedWindow('Camera Control', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Control', cam_width, cam_height)
cv2.moveWindow('Camera Control', 50, 50)

# Создаем окно для указателя (такого же размера, справа)
cv2.namedWindow('Gaze Pointer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gaze Pointer', cam_width, cam_height)
cv2.moveWindow('Gaze Pointer', cam_width + 100, 50)

print("\n  RUNNING")
print("=" * 60)
print("CONTROLS in Camera window:")
print("  'q' - quit")
print("  'c' - calibration mode")
print("  'space' - add calibration point")
print("  'r' - reset")
print("=" * 60)

frame_count = 0
fps_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)

    current_rel = None

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            if len(face_landmarks) > RIGHT_PUPIL:
                left_pupil = face_landmarks[LEFT_PUPIL]
                right_pupil = face_landmarks[RIGHT_PUPIL]

                lx, ly = int(left_pupil.x * w), int(left_pupil.y * h)
                rx, ry = int(right_pupil.x * w), int(right_pupil.y * h)

                cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)
                cv2.circle(frame, (rx, ry), 3, (0, 0, 255), -1)

                left_center, left_size = get_eye_center_and_size(face_landmarks, LEFT_EYE_INDICES, (h, w))
                right_center, right_size = get_eye_center_and_size(face_landmarks, RIGHT_EYE_INDICES, (h, w))

                left_rel = get_relative_pupil((lx, ly), left_center, left_size)
                right_rel = get_relative_pupil((rx, ry), right_center, right_size)

                if left_rel and right_rel:
                    current_rel = ((left_rel[0] + right_rel[0]) / 2,
                                   (left_rel[1] + right_rel[1]) / 2)

                    raw_x, raw_y = rel_to_screen(current_rel[0], current_rel[1])
                    filtered_x, filtered_y = apply_filters(raw_x, raw_y)
                    current_gaze = (filtered_x, filtered_y)

    # Окно указателя (подстраиваем под разрешение)
    pointer_screen = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

    if calibration_mode:
        # Масштабируем калибровочные точки для окна
        scale_x = cam_width / screen_width
        scale_y = cam_height / screen_height

        for i, (x, y) in enumerate(calib_positions):
            wx = int(x * scale_x)
            wy = int(y * scale_y)

            if i < current_calib_idx:
                cv2.circle(pointer_screen, (wx, wy), 10, (0, 255, 0), -1)
                cv2.putText(pointer_screen, f"{i + 1}", (wx - 8, wy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif i == current_calib_idx:
                cv2.circle(pointer_screen, (wx, wy), 15, (0, 0, 255), -1)
                cv2.putText(pointer_screen, f"POINT {i + 1}", (wx - 50, wy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(pointer_screen, "", (wx - 60, wy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.circle(pointer_screen, (wx, wy), 8, (100, 100, 100), 1)
                cv2.putText(pointer_screen, f"{i + 1}", (wx - 6, wy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        cv2.putText(pointer_screen, f"CALIBRATION: {current_calib_idx}/{len(calib_positions)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        # Масштабируем позицию взгляда для окна
        x, y = current_gaze
        wx = int(x * cam_width / screen_width)
        wy = int(y * cam_height / screen_height)

        wx = max(0, min(cam_width - 1, wx))
        wy = max(0, min(cam_height - 1, wy))

        cv2.circle(pointer_screen, (wx, wy), 15, (0, 255, 255), -1)
        cv2.circle(pointer_screen, (wx, wy), 20, (255, 255, 255), 1)
        cv2.line(pointer_screen, (wx - 20, wy), (wx + 20, wy), (255, 255, 255), 1)
        cv2.line(pointer_screen, (wx, wy - 20), (wx, wy + 20), (255, 255, 255), 1)

        cv2.putText(pointer_screen, "Pointer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Информация на камере
    if calibration_mode and current_rel:
        cv2.putText(frame, f"CALIBRATION: {current_calib_idx}/{len(calib_positions)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"REL: ({current_rel[0]:.2f}, {current_rel[1]:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # FPS
    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Gaze Pointer', pointer_screen)
    cv2.imshow('Camera Control', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_mode = not calibration_mode
        if calibration_mode:
            current_calib_idx = 0
            calibration_points = []
            calibration_targets = []
            x_map = []
            y_map = []
            print("\n CALIBRATION MODE ON")
        else:
            print("\n CALIBRATION MODE OFF")
    elif key == ord(' ') and calibration_mode and current_rel:
        if current_calib_idx < len(calib_positions):
            target_x, target_y = calib_positions[current_calib_idx]

            calibration_points.append(current_rel)
            calibration_targets.append((target_x, target_y))
            x_map.append((current_rel[0], target_x))
            y_map.append((current_rel[1], target_y))

            print(f"   Point {current_calib_idx + 1}: ({target_x}, {target_y})")

            current_calib_idx += 1
            if current_calib_idx >= len(calib_positions):
                calibration_mode = False
                print("\n CALIBRATION COMPLETE!")
    elif key == ord('r'):
        calibration_mode = False
        calibration_points = []
        calibration_targets = []
        x_map = []
        y_map = []
        median_x.clear()
        median_y.clear()
        avg_x.clear()
        avg_y.clear()
        lowpass.x = None
        lowpass.y = None
        last_x = None
        last_y = None
        print("\n Reset")

cap.release()
cv2.destroyAllWindows()
print("\n Program finished")