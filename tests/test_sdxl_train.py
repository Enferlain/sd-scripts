import os
import shutil
import subprocess

def test_sdxl_train_dry_run():
    """
    Tests the sdxl_finetune.py script with the dry_run flag to ensure
    that the training setup completes without errors.
    """
    script_path = "scripts/sdxl_finetune.py"
    model_path = "tests/assets/dummy_model.safetensors"
    data_dir = "tests/assets"
    output_dir = "tmp_output"

    # Create a subdirectory for the test data
    test_data_dir = os.path.join(data_dir, "32_test")
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Create dummy data directly in the subdirectory
    for i in range(1, 6):
        with open(os.path.join(test_data_dir, f"test-{i}.latent"), "w") as f:
            f.write("")
        with open(os.path.join(test_data_dir, f"test-{i}.txt"), "w") as f:
            f.write(f"caption for test {i}")

    # Ensure the assets exist
    assert os.path.exists(model_path), f"Dummy model not found at {model_path}"
    assert os.path.exists(data_dir), f"Data directory not found at {data_dir}"

    command = [
        "accelerate", "launch",
        "--num_processes=1",
        script_path,
        "+training.dry_run=True",
        f"sd_models.pretrained_model_name_or_path={model_path}",
        f"dataset.train_data_dir={data_dir}",
        "dataset.resolution=[256,256]",
        "training.max_train_steps=1",
            f"saving.output_dir={output_dir}",
            "saving.output_name=dry_run_test",
        "dataset.reg_data_dir=tests/assets/reg"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    # Print output for debugging, especially on failure
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, "The script failed to execute."
    # The success message is printed to stdout
    assert "Dry run completed successfully." in result.stdout

    # Clean up the test data
    shutil.rmtree(test_data_dir)
    shutil.rmtree(output_dir)
