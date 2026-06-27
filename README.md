# Traffic Analyzer

**English** · [Українською](#аналізатор-трафіку)

Real-time vehicle traffic analysis from an RGB camera stream (RTSP, video file,
or webcam). For every vehicle in view it assigns a stable id, estimates the body
colour, and reads the licence plate, drawing each box in the car's own colour
with an `id | colour | plate` label.

## Features

- **Detection + tracking** — YOLO11n with ByteTrack gives each vehicle a stable
  unique id as it enters and moves through the scene.
- **Colour estimation** — a small MobileNet classifier (`config.color.backend`),
  distilled from a CLIP teacher so it runs at edge speed; an HSV heuristic is
  kept as a fallback backend.
- **Licence plate reading** — a two-stage ONNX pipeline (plate localization +
  OCR) with per-track voting across frames for a stable result. Ukrainian-format
  normalization is available via `config.plate.format_mode`.
- **Built for realtime / edge** — threaded capture, work done on vehicle crops,
  OCR throttled per track, a motion gate that skips static frames, and every
  model running through ONNX Runtime. No torch / ultralytics at runtime, so the
  install is ~250 MB instead of ~2 GB.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All ONNX models are bundled in `weights/`, so nothing else is downloaded.
(`requirements-dev.txt` covers retraining the colour model.)

## Run

```bash
python main.py rtsp://user:pass@host:554/stream   # camera
python main.py weights\test_plates.mp4            # test video file
python main.py 0                                  # webcam
```

Flags: `--skip-empty` (drop frames with no vehicles), `--no-display` (headless),
`--no-motion-filter` (process every frame). Quit with `q` or `Esc`.

## Layout

```
main.py                  open source -> pipeline -> display loop
config.py                tunable parameters per component
core/
  capture.py             video source (threaded live, sequential file)
  pipeline.py            per-frame orchestration + per-track caching
detection/
  vehicle_tracker.py     detector + tracker -> tracks with ids
  yolo_onnx.py           YOLO detector on onnxruntime (no torch)
  byte_track.py          ByteTrack-style tracker (numpy)
  color_estimator.py     colour name + box colour (cnn or heuristic backend)
  color_classifier.py    MobileNet colour model (ONNX)
  plate_detector.py      locate the plate in a vehicle crop
  plate_ocr.py           read text from a plate crop
  plate_reader.py        detect -> OCR -> format
domain/track.py          Track model + colour/plate voting
render/visualizer.py     boxes, labels, FPS overlay
utils/
  frame_filter.py        motion gate + usefulness predicates
  ua_plate.py            plate normalization / UA validation
tools/                   colour dataset generation + model training (dev only)
```

## Approach & design decisions

What was done, and why each approach was chosen:

- **Detection + tracking in one step** — `YOLO.track()` (YOLO11n + ByteTrack)
  detects vehicles and assigns stable ids in a single call, so "spot new cars"
  and "unique id" are handled together. The nano model was picked for realtime.
- **Two-stage plate reading** — a plate is small, so OCR over the whole car is
  unreliable; we localize the plate inside the vehicle crop first, then OCR only
  that. A single frame often misreads, so readings are voted across frames per
  track and the result stays stable.
- **Colour by a distilled CNN** — an HSV heuristic was tried first but kept
  failing on glare and lighting cast (silver cars reading as blue at dusk), since
  a fixed rule can't adapt. A CLIP model classified colour well but is too heavy
  for edge (~1.7 GB), so it was *distilled*: CLIP auto-labelled crops from the
  input videos, the labels were hand-corrected, and a ~6 MB MobileNet was trained
  on them (`tools/`, `requirements-dev.txt` to retrain). Colour is voted across
  frames weighted by crop size, so the answer comes from the closest, clearest
  view. White/silver/gray stays the hardest distinction.
- **Edge / realtime** — every model runs as ONNX through onnxruntime; heavy work
  runs on small vehicle crops, throttled and cached per track; a cheap motion
  gate skips the detector on static frames; and capture runs on a background
  thread that always serves the freshest frame so latency stays low. The vehicle
  detector (`yolo_onnx.py`) and a small numpy ByteTrack (`byte_track.py`) replace
  ultralytics, so **torch is not needed at runtime** — the install drops from
  ~2 GB to ~250 MB. The backend is a one-line swap (ONNX → TensorRT/OpenVINO).

---

# Аналізатор трафіку

[English](#traffic-analyzer) · **Українською**

Аналіз автомобільного трафіку в реальному часі з RGB-відеопотоку (RTSP,
відеофайл або вебкамера). Для кожного авто в кадрі програма присвоює стабільний
ID, визначає колір кузова та зчитує номерний знак, обводячи машину рамкою її
власного кольору з підписом `ID | колір | номер`.

## Можливості

- **Детекція + трекінг** — YOLO11n з ByteTrack дає кожному авто стабільний
  унікальний ID від моменту появи в кадрі.
- **Визначення кольору** — невелика модель MobileNet (`config.color.backend`),
  дистильована з «вчителя» CLIP, тому працює на edge-швидкості; HSV-евристика
  лишається запасним backend-ом.
- **Зчитування номерів** — двоступеневий ONNX-конвеєр (локалізація номера + OCR)
  з голосуванням по кадрах для стабільного результату. Нормалізація під
  український формат — через `config.plate.format_mode`.
- **Розраховано на realtime / edge** — захоплення в окремому потоці, обробка по
  кропах авто, OCR з обмеженою частотою, motion-фільтр пропускає статичні кадри,
  усі моделі працюють через ONNX Runtime. У рантаймі немає torch / ultralytics,
  тож встановлення важить ~250 МБ замість ~2 ГБ.

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Усі ONNX-моделі вже лежать у `weights/`, тож нічого додатково не завантажується.
(`requirements-dev.txt` потрібен лише для перенавчання моделі кольору.)

## Запуск

```bash
python main.py rtsp://user:pass@host:554/stream   # камера
python main.py weights\test_plates.mp4            # тестовий відеофайл
python main.py 0                                  # вебкамера
```

Прапорці: `--skip-empty` (пропускати кадри без авто), `--no-display` (без вікна),
`--no-motion-filter` (обробляти кожен кадр). Вихід — `q` або `Esc`.

## Структура

```
main.py                  точка входу: джерело -> конвеєр -> вікно
config.py                параметри, що налаштовуються, по компонентах
core/
  capture.py             джерело відео (потоково для live, послідовно для файлу)
  pipeline.py            оркестрація по кадрах + кеш по треках
detection/
  vehicle_tracker.py     детектор + трекер -> треки з ID
  yolo_onnx.py           YOLO-детектор на onnxruntime (без torch)
  byte_track.py          трекер у стилі ByteTrack (numpy)
  color_estimator.py     назва кольору + колір рамки (backend cnn або heuristic)
  color_classifier.py    модель кольору MobileNet (ONNX)
  plate_detector.py      пошук номера в кропі авто
  plate_ocr.py           зчитування тексту з кропа номера
  plate_reader.py        детекція -> OCR -> формат
domain/track.py          модель Track + голосування кольору/номера
render/visualizer.py     рамки, підписи, FPS
utils/
  frame_filter.py        motion-фільтр + предикати корисності кадру
  ua_plate.py            нормалізація номера / валідація UA
tools/                   генерація датасету кольору + навчання моделі (для розробки)
```

## Підхід і обґрунтування рішень

Які кроки виконано і чому обрано саме такі підходи:

- **Детекція і трекінг одним кроком** — `YOLO.track()` (YOLO11n + ByteTrack)
  детектує авто та присвоює стабільні ID за один виклик, тож «виявлення нових
  авто» і «унікальний ID» вирішуються разом. Модель nano обрано заради realtime.
- **Двоступеневе зчитування номера** — номер малий, тому OCR по всій машині
  ненадійний; спершу локалізуємо номер у кропі авто, потім розпізнаємо лише його.
  Один кадр часто читається з помилкою, тому результати голосуються по кадрах у
  межах треку — підсумковий номер стабільний.
- **Колір через дистильований CNN** — спершу пробували HSV-евристику, але вона
  ламалася на бліках і кольоровому відтінку освітлення (срібляста машина ввечері
  читалась як синя), бо фіксоване правило не адаптується. Модель CLIP визначала
  колір добре, але занадто важка для edge (~1.7 ГБ), тож її *дистилювали*: CLIP
  авто-розмітив кропи з вхідних відео, мітки виправили вручну, і на них навчили
  MobileNet (~6 МБ) (`tools/`, для перенавчання — `requirements-dev.txt`). Колір
  голосується по кадрах із вагою за розміром кропа, тому відповідь береться з
  найближчого, найчіткішого вигляду. Білий/срібний/сірий лишається найскладнішим
  розрізненням.
- **Edge / realtime** — усі моделі працюють як ONNX через onnxruntime; важка
  робота виконується по малих кропах авто, з обмеженою частотою та кешуванням по
  треках; дешевий motion-фільтр пропускає детектор на статичних кадрах; захоплення
  йде в окремому потоці, що завжди віддає найсвіжіший кадр. Детектор авто
  (`yolo_onnx.py`) і невеликий ByteTrack на numpy (`byte_track.py`) замінюють
  ultralytics, тож **torch у рантаймі не потрібен** — встановлення зменшується з
  ~2 ГБ до ~250 МБ. Backend змінюється одним рядком (ONNX → TensorRT/OpenVINO).
