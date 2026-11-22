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

### Обработка на NVIDIA GPU

Для работы на видеокарте необходим актуальный драйвер NVIDIA и версия PyTorch
с поддержкой CUDA. Отдельно устанавливать CUDA Toolkit не требуется: нужные
библиотеки CUDA входят в пакет PyTorch.

Для GTX 1080 Ti используйте сборку PyTorch для CUDA 11.8. Выполняйте команды
в активированном виртуальном окружении:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Проверьте, что PyTorch действительно обнаружил видеокарту:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Ожидаемый результат для компьютера с GTX 1080 Ti:

```text
CUDA available: True
GPU: NVIDIA GeForce GTX 1080 Ti
```

Если показано `CUDA available: False`, программа будет работать на процессоре.
Проверьте драйвер NVIDIA, активированное виртуальное окружение и установленную
версию PyTorch. При доступной CUDA библиотека Ultralytics автоматически
использует видеокарту, поэтому отдельный аргумент запуска программы не нужен.

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
| `source` | Да | Путь к видеофайлу или URL сетевой трансляции |
| `--output` | Нет | Путь для сохранения обработанного видео |
| `--output-segment-minutes` | Нет | Длительность сегмента потокового видео; по умолчанию 60 минут |
| `--output-max-segments` | Нет | Число хранимых сегментов потока; по умолчанию 24 |
| `--show` | Нет | Показывать обработанные кадры во время работы |
| `--verbose` | Нет | Выводить подробный журнал событий вместо однострочного прогресса |
| `--images-dir` | Нет | Каталог для лучших снимков подтверждённых машин; по умолчанию `results/vehicle_images` |
| `--log-file` | Нет | Файл подробного журнала; по умолчанию `results/vehicle_recognition.log` |
| `--save-to-database` | Нет | Сохранить запуск и подтверждённые наблюдения в MySQL |
| `--camera-name` | Для потока с БД | Имя сетевой камеры; связывает её последовательные запуски в MySQL |
| `--max-frames` | Нет | Остановить обработку после указанного числа кадров |

При использовании `--show` нажмите `q`, чтобы досрочно остановить обработку.
Работа без показа кадров обычно выполняется быстрее.
При сетевом обрыве программа продолжает попытки переподключения без ограничения,
увеличивая паузу между ними до 60 секунд. При использовании `--output` поток
записывается часовыми сегментами; по умолчанию сохраняются последние 24 файла.

При временной недоступности MySQL операции сохранения также повторяются с
увеличением паузы до 60 секунд. После восстановления базы обработка продолжается
без повторного создания уже записанных наблюдений.

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

Подробный журнал всегда записывается в файл, даже без `--verbose`. При достижении
10 МБ журнал переносится в резервный файл; сохраняются пять предыдущих частей,
поэтому длительная работа не приводит к неограниченному росту логов.

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

Для сетевой камеры дополнительно задайте постоянное понятное имя:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8" --save-to-database --camera-name "City camera 1"
```

При первом запуске камера будет создана в таблице `cameras`. Последующие
запуски с тем же именем получат прежний `camera_id`, даже если URL потока
изменился.

Файл `.env` используется одновременно Docker Compose и Python-приложением.
При успешной обработке база получает запись запуска, подтверждённые наблюдения
машин и все временные ID трекера, которые были объединены в каждую машину.
Во время трансляции завершённые и подтверждённые машины сохраняются сразу, не
дожидаясь остановки всего процесса. Повторная запись того же `vehicle_id`
обновляет существующее наблюдение и не создаёт дубликат.

Программа также периодически сохраняет текущий номер кадра и число найденных
машин. Если прежний процесс аварийно завершился и оставил запуск со статусом
`running`, следующий запуск приложения переведёт его в `failed`, сохранив уже
записанные наблюдения и доступный прогресс.

Срок хранения завершённых результатов задаётся в `.env`, например:

```dotenv
DATABASE_RETENTION_DAYS=30
```

При каждом запуске с `--save-to-database`, а затем каждые 24 часа непрерывной
работы результаты старше 30 дней будут удаляться автоматически. Значение `0`
отключает автоматическую очистку. Активные запуски и записи камер не удаляются.

Чтобы удалить результаты обработки, сохранив записи камер, выполните:

```powershell
python -m vehicle_recognition.database_admin clear-results
```

Для защиты от случайного удаления команда потребует ввести `CLEAR`. Она не
выполнится, пока в базе есть активный запуск со статусом `running`.
Вместе со строками наблюдений удаляются связанные снимки внутри
`results/vehicle_images`. Для другого каталога передайте `--images-dir`.

Разовая очистка только устаревших результатов выполняется так:

```powershell
python -m vehicle_recognition.database_admin clear-old-results --days 30
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

### NVIDIA GPU acceleration

GPU processing requires a current NVIDIA driver and a CUDA-enabled PyTorch
build. A separate CUDA Toolkit installation is not required because the
PyTorch package includes the required CUDA runtime libraries.

For a GTX 1080 Ti, install the PyTorch build for CUDA 11.8 inside the activated
virtual environment:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify that PyTorch can access the GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output on the GTX 1080 Ti computer:

```text
CUDA available: True
GPU: NVIDIA GeForce GTX 1080 Ti
```

If `CUDA available: False` is printed, processing will use the CPU. Check the
NVIDIA driver, the active virtual environment, and the installed PyTorch build.
Ultralytics automatically uses an available CUDA GPU, so the application does
not require an additional command-line option.

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

The detailed log rotates at 10 MB and keeps five backup files, preventing
unlimited disk usage during continuous processing.

| Argument | Required | Description |
|---|---:|---|
| `source` | Yes | Path to an input video or a network-stream URL |
| `--output` | No | Path for the annotated output video |
| `--output-segment-minutes` | No | Stream video segment duration; defaults to 60 minutes |
| `--output-max-segments` | No | Number of retained stream segments; defaults to 24 |
| `--show` | No | Display processed frames while running |
| `--verbose` | No | Print detailed tracking events instead of the one-line progress display |
| `--images-dir` | No | Directory for the best confirmed-vehicle images; defaults to `results/vehicle_images` |
| `--log-file` | No | Detailed processing log; defaults to `results/vehicle_recognition.log` |
| `--save-to-database` | No | Save the run and confirmed observations to MySQL |
| `--camera-name` | For a database-backed stream | Stable camera name used to associate its MySQL run history |
| `--max-frames` | No | Stop processing after the specified number of frames |

Press `q` to stop early when `--show` is enabled. Processing without the live
preview is usually faster. The application keeps reconnecting after a stream
interruption, increasing the delay up to 60 seconds. With `--output`, streams
are stored in hourly segments and the latest 24 files are retained by default.

MySQL operations are also retried with a delay of up to 60 seconds during a
temporary outage. Processing resumes after the database becomes available.

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

Provide a stable camera name when saving a network stream:

```powershell
python -m vehicle_recognition.main "https://example.com/live/playlist.m3u8" --save-to-database --camera-name "City camera 1"
```

The first run creates the camera record. Later runs with the same name reuse
its `camera_id` and update its stream URL if necessary.

Finalized stream observations are persisted while the run is still active.
Saving the same `vehicle_id` again updates its existing observation instead of
creating a duplicate.

The application also stores the current frame and accepted-observation count
periodically. On startup, it marks runs left in the `running` state by an
interrupted previous process as `failed` while preserving saved observations
and available progress.

Delete processing results while preserving camera records:

```powershell
python -m vehicle_recognition.database_admin clear-results
```

Associated images inside `results/vehicle_images` are deleted as well. Pass
`--images-dir` when the images were stored in another directory.

Set the automatic retention period in `.env` (`0` disables it):

```dotenv
DATABASE_RETENTION_DAYS=30
```

Retention runs when the application starts and then every 24 hours during
continuous processing. Active runs and camera records are never removed.

Delete only completed results older than a selected number of days:

```powershell
python -m vehicle_recognition.database_admin clear-old-results --days 30
```

Local videos, model weights, and generated results are excluded from the
repository.
