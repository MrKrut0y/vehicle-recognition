from vehicle_recognition.image_cleanup import delete_result_images


def test_delete_result_images_removes_files_and_empty_run_directory(tmp_path):
    image_root = tmp_path / "images"
    run_directory = image_root / "run_1"
    run_directory.mkdir(parents=True)
    image_path = run_directory / "vehicle.jpg"
    image_path.write_bytes(b"image")

    deleted_count = delete_result_images([image_path], image_root)

    assert deleted_count == 1
    assert image_path.exists() is False
    assert run_directory.exists() is False


def test_delete_result_images_preserves_paths_outside_result_directory(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside_path = tmp_path / "important.jpg"
    outside_path.write_bytes(b"keep")

    deleted_count = delete_result_images([outside_path], image_root)

    assert deleted_count == 0
    assert outside_path.exists() is True
