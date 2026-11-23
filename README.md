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

### Требования и установка

Проверенные версии Python: 3.13 и 3.14, 64-bit. Для обычной обработки видео
Docker не нужен. Docker Desktop требуется только для локального запуска MySQL
и использования аргумента `--save-to-database`.

Клонируйте проект и перейдите в его каталог:

```powershell
git clone https://github.com/MrKrut0y/vehicle-recognition.git
cd vehicle-recognition
```

Создайте и активируйте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если PowerShell запрещает активацию окружения, разрешите её только для текущего
окна и повторите команду активации:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Инструменты форматирования и тестирования устанавливаются отдельно:

```powershell
python -m pip install -r requirements-dev.txt
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

Ожидаемый результат:

```text
CUDA available: True
GPU: <название видеокарты NVIDIA>
```

Если показано `CUDA available: False`, программа будет работать на процессоре.
Проверьте драйвер NVIDIA, активированное виртуальное окружение и установленную
версию PyTorch. При доступной CUDA библиотека Ultralytics автоматически
использует видеокарту, поэтому отдельный аргумент запуска программы не нужен.

### Запуск

Для запуска без Docker и базы данных не указывайте `--save-to-database` и
`--camera-name`. Файл `.env` в таком режиме не нужен.

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

После установки `requirements-dev.txt` запустите все автоматические тесты:

```powershell
python -m pytest
```

### Частые проблемы

- `Missing database settings` означает, что для режима с
  `--save-to-database` не найден файл `.env` или в нём не заполнены обязательные
  параметры MySQL. Файл должен находиться в корне проекта рядом с
  `compose.yaml`.
- Если PowerShell не находит команду `docker`, запустите Docker Desktop,
  дождитесь готовности Docker Engine и откройте новое окно PowerShell.
- Если проверка PyTorch выводит `CUDA available: False`, приложение работает на
  процессоре. Проверьте активное виртуальное окружение и переустановите сборку
  PyTorch с CUDA по инструкции выше.
- URL трансляции передаётся как обычная строка вида
  `"https://example.com/live/playlist.m3u8"`. Не вставляйте Markdown-запись
  `[адрес](адрес)` и не добавляйте обратные слеши перед `:` или `.`.
- Отдельные сообщения FFmpeg об ошибках H.264, TLS или TCP обычно означают
  временную потерю пакетов или соединения. Программа повторно подключается к
  потоку автоматически; постоянные повторения требуют проверки сети и
  доступности самой камеры.

### Локальная база данных

Для запуска MySQL 8.4 нужны Docker Desktop и локальный файл `.env`. Docker
Desktop для Windows можно установить командой:

```powershell
winget install --exact --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
```

После установки запустите Docker Desktop и дождитесь готовности Docker Engine.
Для варианта с WSL 2 должны быть включены компоненты Windows Subsystem for Linux
и Virtual Machine Platform. Подробная официальная инструкция:
<https://docs.docker.com/desktop/setup/install/windows-install/>.

Файл `.env` содержит локальные пароли и намеренно не хранится в Git. На каждом
новом компьютере создайте его из шаблона и откройте для редактирования:

```powershell
Copy-Item .env.example .env
notepad .env
```

Обязательно замените значения `MYSQL_PASSWORD` и `MYSQL_ROOT_PASSWORD`:

```dotenv
DB_HOST=127.0.0.1
MYSQL_DATABASE=vehicle_recognition
MYSQL_USER=vehicle_app
MYSQL_PASSWORD=your_application_password
MYSQL_ROOT_PASSWORD=your_root_password
MYSQL_PORT=3306
DATABASE_RETENTION_DAYS=30
```

Не добавляйте `.env` в Git. После сохранения файла запустите MySQL:

```powershell
docker compose up -d
docker compose ps
```

Перед запуском приложения дождитесь состояния `healthy` у контейнера
`vehicle-recognition-mysql`. Если команда `docker` не найдена, запустите Docker
Desktop и заново откройте PowerShell.

Сервер доступен только на этом компьютере по адресу `127.0.0.1:3306`.
Остановка сервера без удаления данных:

```powershell
docker compose down
```

Команда `docker compose down -v` дополнительно удаляет volume `mysql_data` и
всю базу. Не используйте `-v`, если данные нужно сохранить.

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
Если приложение сообщает `Missing database settings`, файл `.env` отсутствует,
сохранён не в корне проекта либо в нём не заполнены обязательные переменные.
Пароли из `.env` создают пользователей только при первом создании пустого
volume; последующее изменение файла не меняет пароль в существующей базе.
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

### Requirements and installation

Tested Python versions are 3.13 and 3.14, 64-bit. Docker is not required for
ordinary video processing. Docker Desktop is only needed to run the local
MySQL server and use `--save-to-database`.

Clone the repository and enter its directory:

```powershell
git clone https://github.com/MrKrut0y/vehicle-recognition.git
cd vehicle-recognition
```

Create and activate a virtual environment, then install the runtime
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current window only and
activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the separate development dependencies before formatting or testing:

```powershell
python -m pip install -r requirements-dev.txt
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

Expected output:

```text
CUDA available: True
GPU: <NVIDIA GPU name>
```

If `CUDA available: False` is printed, processing will use the CPU. Check the
NVIDIA driver, the active virtual environment, and the installed PyTorch build.
Ultralytics automatically uses an available CUDA GPU, so the application does
not require an additional command-line option.

### Usage

For a run without Docker or MySQL, omit `--save-to-database` and
`--camera-name`. The `.env` file is not needed in this mode.

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
python -m pytest
```

### Common problems

- `Missing database settings` means that `.env` was not found or required
  MySQL values are empty. The file must be in the project root next to
  `compose.yaml`.
- If PowerShell cannot find `docker`, start Docker Desktop, wait for the Docker
  Engine to become ready, and open a new PowerShell window.
- If PyTorch reports `CUDA available: False`, processing uses the CPU. Verify
  the active virtual environment and reinstall the CUDA-enabled PyTorch build
  described above.
- Pass a stream URL as a plain string such as
  `"https://example.com/live/playlist.m3u8"`. Do not paste Markdown syntax such
  as `[URL](URL)` or add backslashes before `:` and `.`.
- Occasional FFmpeg H.264, TLS, or TCP messages normally indicate packet loss
  or a temporary connection failure. The application reconnects
  automatically; continuous errors require checking the network and camera
  availability.

### Local database

Install Docker Desktop on Windows if it is not already available:

```powershell
winget install --exact --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
```

Start Docker Desktop and wait for the Docker Engine. The WSL 2 backend requires
Windows Subsystem for Linux and Virtual Machine Platform. See the
[official Windows installation guide](https://docs.docker.com/desktop/setup/install/windows-install/)
for current requirements.

The local `.env` file contains passwords and is intentionally excluded from
Git. Create it from the template on every new computer and edit its values:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace both example passwords before starting the database:

```dotenv
DB_HOST=127.0.0.1
MYSQL_DATABASE=vehicle_recognition
MYSQL_USER=vehicle_app
MYSQL_PASSWORD=your_application_password
MYSQL_ROOT_PASSWORD=your_root_password
MYSQL_PORT=3306
DATABASE_RETENTION_DAYS=30
```

Do not commit `.env`. Start the MySQL 8.4 development server with Docker
Compose:

```powershell
docker compose up -d
docker compose ps
```

Wait until `vehicle-recognition-mysql` reports `healthy` before starting the
application. If `docker` is not recognized, start Docker Desktop and reopen
PowerShell.

The server is available only on the local machine at `127.0.0.1:3306`.
Initial SQL files from `database/init` are applied automatically when Docker
creates an empty database volume.

Stop the server without deleting its data:

```powershell
docker compose down
```

Adding `-v` to this command also deletes the `mysql_data` volume and the entire
database. Do not use it when the data must be preserved.

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

The same `.env` file is read by Docker Compose and the Python application. The
`Missing database settings` error means that it is missing from the project
root or contains empty required values. MySQL users and passwords are created
only when an empty volume is initialized; changing `.env` later does not alter
passwords inside an existing database.

Delete only completed results older than a selected number of days:

```powershell
python -m vehicle_recognition.database_admin clear-old-results --days 30
```

Local videos, model weights, and generated results are excluded from the
repository.
