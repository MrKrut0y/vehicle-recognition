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

Вместо файла можно передать URL трансляции HLS, RTSP или RTMP:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8"
```

Поток обрабатывается до `Ctrl+C`. Для короткой проверки ограничьте число кадров:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8" --max-frames 60
```

Чтобы сохранить видео с рамками, идентификаторами и текущим количеством
транспорта, укажите выходной файл:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4"
```

Каталог для выходного файла создаётся автоматически. Без параметра `--output`
программа выполняет распознавание и выводит итоговое количество, но не сохраняет
обработанное видео.

В подписи рамки `V` обозначает внутренний ID физической машины, а `T` —
временный ID трекера. После краткого перекрытия `T` может измениться, но при
успешном восстановлении `V` остаётся прежним.

### Аргументы командной строки

| Аргумент | Обязательный | Назначение |
|---|---:|---|
| `source` | Да | Путь к исходному видеофайлу |
| `--output` | Нет | Путь для сохранения обработанного видео |
| `--show` | Нет | Показывать обработанные кадры во время работы |
| `--verbose` | Нет | Выводить подробный журнал событий вместо однострочного прогресса |
| `--images-dir` | Нет | Каталог для лучших снимков подтверждённых машин; по умолчанию `results/vehicle_images` |
| `--log-file` | Нет | Файл подробного журнала; по умолчанию `results/vehicle_recognition.log` |
| `--save-to-database` | Нет | Сохранить запуск и подтверждённые наблюдения в MySQL |
| `--max-frames` | Нет | Остановить обработку после указанного числа кадров |

При использовании `--show` нажмите `q`, чтобы досрочно остановить обработку.
Работа без показа кадров обычно выполняется быстрее.
При кратком сетевом обрыве поток автоматически переподключается. Использование
`--output` без ограничения времени создаёт постоянно растущий видеофайл.

### Режимы вывода

По умолчанию программа обновляет одну строку с процентом обработки, номером
кадра и текущим количеством подтверждённых машин:

```text
Progress: 42.6% | frames 1278/3000 | 6.0 FPS | remaining 04:47 | vehicles 18 | stitched 9
```

Расширенный режим выводит отдельные сообщения о прогрессе с оставшимся временем,
создании кандидатов, подтверждении машин, восстановлении потерянных ID и
объединении дублирующихся треков:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4" --verbose
```

Подробный журнал всегда записывается в файл, даже без `--verbose`. При новом
запуске указанный файл перезаписывается, чтобы события разных тестов не смешивались.

### Тесты

Для запуска всех автоматических тестов выполните:

```powershell
pytest
```

### Локальная база данных

Для запуска MySQL 8.4 нужны Docker Desktop и локальный файл `.env`,
созданный по образцу `.env.example`.

```powershell
docker compose up -d
docker compose ps
```

Сервер доступен только на этом компьютере по адресу `127.0.0.1:3306`.
Остановка сервера без удаления данных:

```powershell
docker compose down
```

На пустой базе Docker автоматически применяет SQL-файлы из `database/init`.
Схема хранит камеры, запуски обработки, подтверждённые наблюдения и временные
ID трекера. Первоначальная схема находится в `database/init/001_initial_schema.sql`.

После запуска контейнера результаты можно сохранить командой:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --save-to-database
```

Файл `.env` используется одновременно Docker Compose и Python-приложением.
При успешной обработке база получает запись запуска, подтверждённые наблюдения
машин и все временные ID трекера, которые были объединены в каждую машину.

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

Process an HLS, RTSP, or RTMP stream until `Ctrl+C` is pressed:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8"
```

Limit a stream test to 60 frames:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8" --max-frames 60
```

Save an annotated video:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --output ".\results\test3_result.mp4"
```

Frame labels use `V` for the canonical physical-vehicle ID and `T` for the
temporary tracker ID. After a short occlusion, `T` can change while a
successfully recovered vehicle keeps the same `V`.

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
| `--images-dir` | No | Directory for the best confirmed-vehicle images; defaults to `results/vehicle_images` |
| `--log-file` | No | Detailed processing log; defaults to `results/vehicle_recognition.log` |
| `--save-to-database` | No | Save the run and confirmed observations to MySQL |
| `--max-frames` | No | Stop processing after the specified number of frames |

Press `q` to stop early when `--show` is enabled. Processing without the live
preview is usually faster. The application attempts to reconnect after a short
stream interruption. Using `--output` without a frame limit creates a video file
that continues to grow until processing stops.

### Tests

```powershell
pytest
```

### Local database

Start the MySQL 8.4 development server with Docker Compose:

```powershell
docker compose up -d
docker compose ps
```

The server is available only on the local machine at `127.0.0.1:3306`.
Initial SQL files from `database/init` are applied automatically when Docker
creates an empty database volume.

Save processing results after the database container becomes healthy:

```powershell
python -m vehicle_recognition.main ".\videos\test3.mp4" --save-to-database
```

Local videos, model weights, and generated results are excluded from the
repository.
