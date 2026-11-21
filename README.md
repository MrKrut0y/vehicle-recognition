# Распознавание транспорта / Vehicle Recognition

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

### Запуск

Команда запуска принимает обязательный путь к исходному видео:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4"
```

Чтобы сохранить видео с рамками, идентификаторами и текущим количеством
транспорта, укажите выходной файл:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4"
```

Каталог для выходного файла создаётся автоматически. Без параметра `--output`
программа выполняет распознавание и выводит итоговое количество, но не сохраняет
обработанное видео.

### Аргументы командной строки

| Аргумент | Обязательный | Назначение |
|---|---:|---|
| `source` | Да | Путь к исходному видеофайлу |
| `--output` | Нет | Путь для сохранения обработанного видео |
| `--show` | Нет | Показывать обработанные кадры во время работы |
| `--verbose` | Нет | Выводить подробный журнал событий вместо однострочного прогресса |

При использовании `--show` нажмите `q`, чтобы досрочно остановить обработку.
Работа без показа кадров обычно выполняется быстрее.

### Режимы вывода

По умолчанию программа обновляет одну строку с процентом обработки, номером
кадра и текущим количеством подтверждённых машин:

```text
Обработано: 42.60% | Кадр: 1278/3000 | Машин: 18
```

Расширенный режим выводит отдельные сообщения о создании кандидатов,
подтверждении машин, восстановлении потерянных ID и объединении дублирующихся
треков:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4" --verbose
```

### Тесты

Для запуска всех автоматических тестов выполните:

```powershell
pytest
```

Локальные видеозаписи, веса моделей и результаты обработки не добавляются в
репозиторий. Их расположение задаётся пользователем при запуске.

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

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Usage

Process a video without saving annotated output:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4"
```

Save an annotated video:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4"
```

Enable detailed event logs:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4" --verbose
```

| Argument | Required | Description |
|---|---:|---|
| `source` | Yes | Path to the input video |
| `--output` | No | Path for the annotated output video |
| `--show` | No | Display processed frames while running |
| `--verbose` | No | Print detailed tracking events instead of the one-line progress display |

Press `q` to stop early when `--show` is enabled. Processing without the live
preview is usually faster.

### Tests

```powershell
pytest
```

Local videos, model weights, and generated results are excluded from the
repository.
