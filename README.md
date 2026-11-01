# Распознавание транспорта в Глазове / Glazov Vehicle Recognition

## Русский

Курсовой проект по автоматическому заполнению базы данных сведениями о транспортных средствах, обнаруженных на видеозаписи городской камеры средствами компьютерного зрения.

### Цель проекта

Разработать систему, которая получает видеопоток одной камеры в городе Глазове, обнаруживает и отслеживает транспортные средства, определяет доступные признаки и сохраняет результаты в базе данных MySQL.

### Планируемая работа системы

1. Получение разрешённого видеопотока или видеозаписи одной камеры в Глазове.
2. Обнаружение и отслеживание транспорта с помощью YOLOv8.
3. Определение типа транспортного средства.
4. Распознавание государственного регистрационного номера и цвета, если качество изображения позволяет это сделать.
5. Создание одной записи в MySQL для каждого учтённого транспортного средства.
6. Отметка результатов с низкой достоверностью для последующей ручной проверки.

### Запуск YOLO

Общая структура используемой команды:

```powershell
yolo РЕЖИМ model=МОДЕЛЬ source=ИСТОЧНИК classes=КЛАССЫ tracker=ТРЕКЕР conf=ПОРОГ save=True name=ИМЯ
```

Нужные для проекта параметры:

- `РЕЖИМ` — `predict` для обнаружения объектов на изображении или `track` для обнаружения и отслеживания транспорта на видео;
- `model=yolov8s.pt` — готовая модель YOLOv8s;
- `source=...` — путь к изображению, видеофайлу или разрешённому видеопотоку;
- `classes=2,3,5,7` — обрабатывать только легковые автомобили, мотоциклы, автобусы и грузовые автомобили; люди исключаются;
- `tracker=botsort.yaml` — отслеживание на записи с движущейся камеры;
- `tracker=bytetrack.yaml` — простой вариант для будущей неподвижной городской камеры;
- `conf=0.25` — минимальная уверенность обнаружения от 0 до 1; объекты с меньшим значением отбрасываются;
- `save=True` — сохранить обработанный материал;
- `name=...` — имя папки с результатом запуска.

Пример обнаружения транспорта на изображении:

```powershell
yolo predict model=yolov8s.pt source=image.jpg classes=2,3,5,7 conf=0.25 save=True name=image_test
```

Пример отслеживания транспорта на видео с регистратора:

```powershell
yolo track model=yolov8s.pt source=test.mp4 classes=2,3,5,7 tracker=botsort.yaml conf=0.25 save=True name=dashcam_test
```

### Текущее состояние

- Выполнен анализ предметной области.
- Подготовлена первоначальная структура базы данных MySQL.
- Рассмотрены операторы городских камер в Глазове.
- Для первого варианта выбрана нейросетевая модель YOLOv8.
- Найдены дополнительные модели для распознавания российских регистрационных номеров и цвета автомобиля.
- Подготовлена локальная среда Python и установлена библиотека Ultralytics.
- Следующий этап — проверить YOLOv8 на тестовом изображении, а затем на короткой видеозаписи дорожного движения.

### Содержимое репозитория

- `Reports/Курсовая.docx` — текущий текст курсовой работы;
- `db.sql` — первоначальный SQL-скрипт создания базы данных MySQL;
- `Reports/Отчёт_о_ходе_работы.md` — журнал исследования и принятых решений;
- `Reports/Гайд_YOLO_первый_запуск.md` — инструкция по первому запуску YOLOv8.

Виртуальная среда Python, файлы моделей, тестовые видеозаписи, результаты обработки и предоставленные методические материалы не загружаются в репозиторий.

## English

This coursework project focuses on automatically populating a database with information about vehicles detected in footage from a city camera using computer vision.

### Project goal

Develop a system that receives video from one camera in Glazov, detects and tracks vehicles, recognizes available vehicle attributes, and stores the results in a MySQL database.

### Planned workflow

1. Receive an authorized video stream or recording from one camera in Glazov.
2. Detect and track vehicles with YOLOv8.
3. Determine the vehicle type.
4. Recognize the registration plate and vehicle color when the image quality is sufficient.
5. Create one MySQL observation record for each counted vehicle.
6. Mark low-confidence results for manual review.

### Running YOLO

General structure of the command used in this project:

```powershell
yolo MODE model=MODEL source=SOURCE classes=CLASSES tracker=TRACKER conf=THRESHOLD save=True name=NAME
```

Parameters required by the project:

- `MODE` — use `predict` to detect objects in an image or `track` to detect and track vehicles in a video;
- `model=yolov8s.pt` — use the pretrained YOLOv8s model;
- `source=...` — specify an image, video file, or authorized video stream;
- `classes=2,3,5,7` — process only cars, motorcycles, buses, and trucks; people are excluded;
- `tracker=botsort.yaml` — track vehicles in moving-camera footage;
- `tracker=bytetrack.yaml` — use a simple tracker for the future fixed city camera;
- `conf=0.25` — set the minimum detection confidence from 0 to 1; lower-confidence detections are discarded;
- `save=True` — save the processed media;
- `name=...` — set the output run directory name.

Example of vehicle detection in an image:

```powershell
yolo predict model=yolov8s.pt source=image.jpg classes=2,3,5,7 conf=0.25 save=True name=image_test
```

Example of vehicle tracking in dashcam footage:

```powershell
yolo track model=yolov8s.pt source=test.mp4 classes=2,3,5,7 tracker=botsort.yaml conf=0.25 save=True name=dashcam_test
```

### Current status

- The subject area has been analyzed.
- An initial MySQL database structure has been prepared.
- Potential city camera operators in Glazov have been reviewed.
- YOLOv8 has been selected as the primary neural network model.
- Additional models for Russian registration plate recognition and vehicle color classification have been identified.
- A local Python environment has been prepared and the Ultralytics package has been installed.
- The next step is to test YOLOv8 on a sample image and then on a short traffic video.

### Repository contents

- `Reports/Курсовая.docx` — current coursework document;
- `db.sql` — initial MySQL database creation script;
- `Reports/Отчёт_о_ходе_работы.md` — research and decision log;
- `Reports/Гайд_YOLO_первый_запуск.md` — first YOLOv8 launch guide.

The local Python environment, model weights, test videos, generated results, and supplied formatting reference materials are excluded from version control.
