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
