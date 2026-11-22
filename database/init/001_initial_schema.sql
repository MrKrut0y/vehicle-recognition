CREATE TABLE IF NOT EXISTS cameras (
    camera_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(255) NULL,
    stream_url VARCHAR(2048) NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (camera_id),
    CONSTRAINT uq_cameras_name UNIQUE (name)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    camera_id BIGINT UNSIGNED NULL,
    source_type VARCHAR(16) NOT NULL,
    source_name VARCHAR(1024) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'running',
    started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at DATETIME(6) NULL,
    total_frames INT UNSIGNED NULL,
    processed_frames INT UNSIGNED NOT NULL DEFAULT 0,
    source_fps DECIMAL(8, 3) NULL,
    accepted_observations INT UNSIGNED NOT NULL DEFAULT 0,
    error_message TEXT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (run_id),
    CONSTRAINT fk_processing_runs_camera
        FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
        ON UPDATE RESTRICT
        ON DELETE SET NULL,
    CONSTRAINT chk_processing_runs_source_type
        CHECK (source_type IN ('video', 'stream')),
    CONSTRAINT chk_processing_runs_status
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_processing_runs_frames
        CHECK (total_frames IS NULL OR processed_frames <= total_frames),
    CONSTRAINT chk_processing_runs_time
        CHECK (finished_at IS NULL OR finished_at >= started_at),
    INDEX idx_processing_runs_camera_started (camera_id, started_at),
    INDEX idx_processing_runs_status (status)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS vehicle_observations (
    observation_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_id BIGINT UNSIGNED NOT NULL,
    vehicle_id INT UNSIGNED NOT NULL,
    vehicle_type VARCHAR(32) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    first_frame INT UNSIGNED NOT NULL,
    last_frame INT UNSIGNED NOT NULL,
    frames_seen INT UNSIGNED NOT NULL,
    first_seen_seconds DECIMAL(12, 3) NOT NULL,
    last_seen_seconds DECIMAL(12, 3) NOT NULL,
    movement_ratio DECIMAL(8, 6) NOT NULL,
    pixel_motion_frames INT UNSIGNED NOT NULL,
    max_pixel_motion_ratio DECIMAL(8, 6) NOT NULL,
    completion_reason VARCHAR(32) NOT NULL,
    image_path VARCHAR(1024) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (observation_id),
    CONSTRAINT uq_vehicle_observations_run_vehicle
        UNIQUE (run_id, vehicle_id),
    CONSTRAINT fk_vehicle_observations_run
        FOREIGN KEY (run_id) REFERENCES processing_runs (run_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT chk_vehicle_observations_type
        CHECK (vehicle_type IN ('car', 'motorcycle', 'bus', 'truck')),
    CONSTRAINT chk_vehicle_observations_confidence
        CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_vehicle_observations_frames
        CHECK (first_frame <= last_frame AND frames_seen > 0),
    CONSTRAINT chk_vehicle_observations_seconds
        CHECK (
            first_seen_seconds >= 0
            AND last_seen_seconds >= first_seen_seconds
        ),
    CONSTRAINT chk_vehicle_observations_movement
        CHECK (movement_ratio >= 0),
    CONSTRAINT chk_vehicle_observations_pixel_motion
        CHECK (
            max_pixel_motion_ratio >= 0
            AND max_pixel_motion_ratio <= 1
        ),
    CONSTRAINT chk_vehicle_observations_completion
        CHECK (completion_reason IN ('track_lost', 'video_ended')),
    INDEX idx_vehicle_observations_run_type (run_id, vehicle_type),
    INDEX idx_vehicle_observations_created (created_at)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS observation_tracker_ids (
    observation_id BIGINT UNSIGNED NOT NULL,
    tracker_id BIGINT UNSIGNED NOT NULL,
    frames_seen INT UNSIGNED NOT NULL,
    PRIMARY KEY (observation_id, tracker_id),
    CONSTRAINT fk_observation_tracker_ids_observation
        FOREIGN KEY (observation_id)
            REFERENCES vehicle_observations (observation_id)
        ON UPDATE RESTRICT
        ON DELETE CASCADE,
    CONSTRAINT chk_observation_tracker_ids_frames
        CHECK (frames_seen > 0),
    INDEX idx_observation_tracker_ids_tracker (tracker_id)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;
