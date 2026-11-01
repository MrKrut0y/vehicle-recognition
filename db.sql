SET NAMES utf8mb4;

CREATE DATABASE IF NOT EXISTS vehicle_monitoring
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE vehicle_monitoring;

CREATE TABLE IF NOT EXISTS cameras (
  camera_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_name VARCHAR(100) NOT NULL,
  city VARCHAR(100) NOT NULL DEFAULT 'Глазов',
  location_description VARCHAR(255) NOT NULL,
  stream_url VARCHAR(500) NULL,
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (camera_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS vehicle_observations (
  observation_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_id BIGINT UNSIGNED NOT NULL,
  observed_at DATETIME(3) NOT NULL,
  vehicle_type VARCHAR(32) NOT NULL,
  plate_number VARCHAR(20) NULL,
  vehicle_color VARCHAR(32) NULL,
  confidence DECIMAL(5,4) NULL,
  track_id VARCHAR(64) NULL,
  frame_path VARCHAR(500) NULL,
  review_status VARCHAR(20) NOT NULL DEFAULT 'new',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (observation_id),
  KEY idx_observations_camera_time (camera_id, observed_at),
  KEY idx_observations_plate (plate_number),
  CONSTRAINT fk_observations_camera
    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB;
