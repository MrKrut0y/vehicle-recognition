# Распознавание транспорта в Глазове / Glazov Vehicle Recognition

## Русский

Учебный курсовой проект по автоматическому обнаружению и учёту транспортных средств на видеозаписях городских камер с использованием YOLOv8.

### Цель

Разработать программу, которая получает видео, обнаруживает и отслеживает транспорт, определяет доступные характеристики и сохраняет результаты наблюдения в базе данных MySQL.

### План разработки

1. Обнаружение транспорта на видео.
2. Отслеживание и подсчёт автомобилей без повторных записей.
3. Определение типа, цвета и государственного номера.
4. Сохранение результатов в MySQL.
5. Подключение разрешённого видеопотока городской камеры.

Проект разрабатывается поэтапно. Каждая функция сначала проверяется на локальной видеозаписи.

### Установка зависимостей

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## English

An educational coursework project for automatic vehicle detection and registration in city-camera footage using YOLOv8.

### Goal

Develop an application that receives video, detects and tracks vehicles, recognizes available attributes, and stores observation results in a MySQL database.

### Development plan

1. Detect vehicles in video.
2. Track and count vehicles without duplicate records.
3. Recognize vehicle type, colour, and registration plate.
4. Store results in MySQL.
5. Connect an authorized city-camera video stream.

The project is developed incrementally. Each feature is first tested on a local video recording.
