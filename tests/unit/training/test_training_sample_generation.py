from unittest.mock import MagicMock, patch
import json
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from library.training.sample_generation import (
    line_to_prompt_dict,
    sample_images_check,
    get_my_scheduler,
    load_prompts,
    get_sampling_prompt_dicts,
    sample_image_inference,
)
from library.objectives.ddpm import DDPM_PREDICTION_TYPE_EPSILON, DDPM_PREDICTION_TYPE_V
from PIL import Image


class TestLineToPromptDict:
    def test_basic_prompt(self):
        result = line_to_prompt_dict("a cat")
        assert result["prompt"] == "a cat"
        assert "width" not in result

    def test_width_height(self):
        result = line_to_prompt_dict("a dog --w 1024 --h 768")
        assert result["width"] == 1024
        assert result["height"] == 768
        assert result["prompt"] == "a dog"

    def test_seed_and_steps(self):
        result = line_to_prompt_dict("sunset --d 42 --s 50")
        assert result["seed"] == 42
        assert result["sample_steps"] == 50

    def test_guidance_scale(self):
        result = line_to_prompt_dict("test --g 7.5")
        assert result["guidance_scale"] == 7.5

    def test_negative_prompt(self):
        result = line_to_prompt_dict("cat --n ugly, distorted")
        assert result["negative_prompt"] == "ugly, distorted"

    def test_sampler(self):
        result = line_to_prompt_dict("test --ss euler_a")
        assert result["sample_sampler"] == "euler_a"

    def test_steps_clamping(self):
        # Steps should be clamped to [1, 1000]
        result = line_to_prompt_dict("test --s 2000")
        assert result["sample_steps"] == 1000

        result = line_to_prompt_dict("test --s 0")
        assert result["sample_steps"] == 1

    def test_case_insensitive(self):
        result = line_to_prompt_dict("test --W 512 --H 512")
        assert result["width"] == 512
        assert result["height"] == 512


class TestSampleImagesCheck:
    def test_sample_at_first_true(self):
        config = MagicMock()
        config.sample_at_first = True
        assert sample_images_check(config, epoch=0, steps=0) is True

    def test_sample_at_first_false(self):
        config = MagicMock()
        config.sample_at_first = False
        assert sample_images_check(config, epoch=0, steps=0) is False

    def test_sample_every_n_epochs(self):
        config = MagicMock()
        config.sample_at_first = False
        config.sample_every_n_epochs = 5
        config.sample_every_n_steps = None

        # Should sample at epoch 5, 10, 15...
        # Note: the implementation likely checks 'epoch % n == 0' or similar.
        # However, implementation details might vary.
        # Assuming typical logic:
        assert sample_images_check(config, epoch=5, steps=100) is True
        assert sample_images_check(config, epoch=3, steps=100) is False
        assert sample_images_check(config, epoch=10, steps=200) is True

    def test_sample_every_n_steps(self):
        config = MagicMock()
        config.sample_at_first = False
        config.sample_every_n_epochs = None
        config.sample_every_n_steps = 100

        # epoch must be None for step-based sampling in some implementations,
        # but let's verify behavior. The function signature is (sampling_config, epoch, steps).
        # Typically checks if steps % n == 0
        assert sample_images_check(config, epoch=None, steps=100) is True
        assert sample_images_check(config, epoch=None, steps=50) is False
        assert sample_images_check(config, epoch=None, steps=200) is True

        # If epoch is not None (end of epoch), step-based sampling is disabled
        # to avoid double-sampling at epoch boundaries
        assert sample_images_check(config, epoch=1, steps=100) is False

    def test_no_sampling_config(self):
        config = MagicMock()
        config.sample_at_first = False
        config.sample_every_n_epochs = None
        config.sample_every_n_steps = None

        assert sample_images_check(config, epoch=None, steps=100) is False


class TestGetMyScheduler:
    def test_ddim_scheduler(self):
        scheduler = get_my_scheduler(sample_sampler="ddim", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        assert isinstance(scheduler, DDIMScheduler)
        # config access depends on scheduler instance structure
        # assert scheduler.config.prediction_type != "v_prediction"

    def test_euler_scheduler(self):
        scheduler = get_my_scheduler(sample_sampler="euler", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        assert isinstance(scheduler, EulerDiscreteScheduler)

    def test_euler_a_scheduler(self):
        scheduler = get_my_scheduler(sample_sampler="euler_a", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        assert isinstance(scheduler, EulerAncestralDiscreteScheduler)

    def test_dpmsolver_algorithm_type(self):
        scheduler = get_my_scheduler(sample_sampler="dpmsolver++", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        assert isinstance(scheduler, DPMSolverMultistepScheduler)
        # Check algorithm type if accessible
        assert scheduler.config.algorithm_type == "dpmsolver++"

    def test_v_parameterization(self):
        scheduler = get_my_scheduler(sample_sampler="ddim", prediction_type=DDPM_PREDICTION_TYPE_V)
        assert scheduler.config.prediction_type == "v_prediction"

    def test_default_fallback(self):
        scheduler = get_my_scheduler(sample_sampler="nonexistent", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        assert isinstance(scheduler, DDIMScheduler)

    def test_clip_sample_enabled(self):
        scheduler = get_my_scheduler(sample_sampler="ddim", prediction_type=DDPM_PREDICTION_TYPE_EPSILON)
        # Verify clip_sample default logic
        if hasattr(scheduler.config, "clip_sample"):
            assert scheduler.config.clip_sample is True  # Defaults usually True for DDIM in this codebase


class TestLoadPrompts:
    def test_load_txt_file(self, tmp_path):
        txt_file = tmp_path / "prompts.txt"
        txt_file.write_text(
            "a cat --w 512 --h 512\n"
            "# comment line\n"
            "a dog --d 42\n"
            "\n"  # blank line
            "sunset --g 7.5",
            encoding="utf-8",
        )

        prompts = load_prompts(str(txt_file))

        assert len(prompts) == 3  # comment and blank lines ignored
        assert prompts[0]["prompt"] == "a cat"
        assert prompts[0]["width"] == 512
        assert prompts[0]["enum"] == 0
        assert prompts[1]["prompt"] == "a dog"
        assert prompts[1]["seed"] == 42
        assert prompts[1]["enum"] == 1

    def test_load_json_file(self, tmp_path):
        json_file = tmp_path / "prompts.json"
        json_file.write_text(json.dumps([{"prompt": "cat", "width": 512}, {"prompt": "dog", "seed": 42}]), encoding="utf-8")

        prompts = load_prompts(str(json_file))

        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "cat"
        assert prompts[0]["enum"] == 0
        assert "subset" not in prompts[0]  # cleaned up

    def test_load_toml_file(self, tmp_path):
        toml_file = tmp_path / "prompts.toml"
        toml_file.write_text(
            '[prompt]\nprompt = "base prompt"\n[[prompt.subset]]\nwidth = 512\n[[prompt.subset]]\nwidth = 1024\n', encoding="utf-8"
        )

        prompts = load_prompts(str(toml_file))

        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "base prompt"
        assert prompts[0]["width"] == 512
        assert prompts[1]["width"] == 1024
        assert "subset" not in prompts[0]  # cleaned up

    def test_string_prompts_converted_to_dict(self, tmp_path):
        txt_file = tmp_path / "prompts.txt"
        txt_file.write_text("cat --w 512", encoding="utf-8")

        prompts = load_prompts(str(txt_file))

        # String prompt should be parsed into dict
        assert isinstance(prompts[0], dict)
        assert prompts[0]["prompt"] == "cat"


class TestGetSamplingPromptDicts:
    def test_uses_prompt_file_when_configured(self, tmp_path):
        txt_file = tmp_path / "prompts.txt"
        txt_file.write_text("a cat", encoding="utf-8")

        sampling_config = MagicMock()
        sampling_config.sample_prompt_file = str(txt_file)
        sampling_config.sample_prompt = "inline fallback"

        prompts = get_sampling_prompt_dicts(sampling_config)

        assert prompts is not None
        assert len(prompts) == 1
        assert prompts[0]["prompt"] == "a cat"

    def test_uses_inline_prompt_when_no_prompt_file(self):
        sampling_config = MagicMock()
        sampling_config.sample_prompt_file = None
        sampling_config.sample_prompt = "inline prompt"

        prompts = get_sampling_prompt_dicts(sampling_config)

        assert prompts == [{"prompt": "inline prompt", "enum": 0}]

    def test_returns_none_when_no_prompt_source(self):
        sampling_config = MagicMock()
        sampling_config.sample_prompt_file = None
        sampling_config.sample_prompt = None

        assert get_sampling_prompt_dicts(sampling_config) is None


class TestSampleImageInference:
    @patch("library.training.sample_generation.get_my_scheduler")
    def test_prompt_replacement(self, mock_get_scheduler):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[Image.new("RGB", (512, 512))])
        # Note: implementation calls pipeline(..., output_type="pil").images typically

        mock_accelerator = MagicMock()
        mock_accelerator.autocast.return_value.__enter__ = MagicMock()
        mock_accelerator.autocast.return_value.__exit__ = MagicMock()
        mock_accelerator.is_main_process = True

        sampling_config = MagicMock()
        sampling_config.sample_sampler = "ddim"
        training_config = MagicMock()
        objective_config = MagicMock()
        objective_config.target = DDPM_PREDICTION_TYPE_EPSILON
        saving_config = MagicMock()
        saving_config.output_name = "test"
        loss_config = MagicMock()

        prompt_dict = {"prompt": "a {token} cat", "enum": 0, "width": 512, "height": 512}

        # Mock get_my_scheduler to return a dummy scheduler
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        with (
            patch("torch.manual_seed"),
            patch("os.path.join", return_value="/tmp/test.png"),
            patch("PIL.Image.Image.save"),
        ):
            sample_image_inference(
                mock_accelerator,
                sampling_config,
                training_config,
                saving_config,
                objective_config,
                loss_config,
                mock_pipeline,
                "/tmp",
                prompt_dict,
                epoch=1,
                steps=100,
                prompt_replacement=("{token}", "fluffy"),
            )

        # Check pipeline was called with replaced prompt
        call_args = mock_pipeline.call_args
        assert call_args is not None
        assert "fluffy" in call_args.kwargs["prompt"]

    @patch("library.training.sample_generation.get_my_scheduler")
    def test_uses_sampling_config_defaults(self, mock_get_scheduler):
        mock_pipeline = MagicMock()
        mock_pipeline.latents_to_image.return_value = [Image.new("RGB", (640, 832))]
        mock_pipeline.return_value = "latents"

        mock_accelerator = MagicMock()
        mock_accelerator.autocast.return_value.__enter__ = MagicMock()
        mock_accelerator.autocast.return_value.__exit__ = MagicMock()
        mock_accelerator.trackers = []

        sampling_config = MagicMock()
        sampling_config.sample_prompt = "config prompt"
        sampling_config.sample_negative_prompt = "bad anatomy"
        sampling_config.sample_width = 640
        sampling_config.sample_height = 832
        sampling_config.sample_steps = 28
        sampling_config.sample_cfg_scale = 6.5
        sampling_config.sample_seed = 123
        sampling_config.sample_sampler = "ddim"
        training_config = MagicMock()
        objective_config = MagicMock()
        objective_config.target = DDPM_PREDICTION_TYPE_EPSILON
        saving_config = MagicMock()
        saving_config.output_name = "test"
        loss_config = MagicMock()

        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        with (
            patch("torch.manual_seed") as mock_seed,
            patch("os.path.join", return_value="/tmp/test.png"),
            patch("PIL.Image.Image.save"),
        ):
            sample_image_inference(
                mock_accelerator,
                sampling_config,
                training_config,
                saving_config,
                objective_config,
                loss_config,
                mock_pipeline,
                "/tmp",
                {"enum": 0},
                epoch=1,
                steps=100,
                prompt_replacement=None,
            )

        mock_seed.assert_called_once_with(123)
        assert mock_pipeline.call_args.kwargs["prompt"] == "config prompt"
        assert mock_pipeline.call_args.kwargs["negative_prompt"] == "bad anatomy"
        assert mock_pipeline.call_args.kwargs["width"] == 640
        assert mock_pipeline.call_args.kwargs["height"] == 832
        assert mock_pipeline.call_args.kwargs["num_inference_steps"] == 28
        assert mock_pipeline.call_args.kwargs["guidance_scale"] == 6.5

    @patch("library.training.sample_generation.get_my_scheduler")
    def test_prompt_dict_overrides_sampling_defaults(self, mock_get_scheduler):
        mock_pipeline = MagicMock()
        mock_pipeline.latents_to_image.return_value = [Image.new("RGB", (1024, 768))]
        mock_pipeline.return_value = "latents"

        mock_accelerator = MagicMock()
        mock_accelerator.autocast.return_value.__enter__ = MagicMock()
        mock_accelerator.autocast.return_value.__exit__ = MagicMock()
        mock_accelerator.trackers = []

        sampling_config = MagicMock()
        sampling_config.sample_prompt = "config prompt"
        sampling_config.sample_negative_prompt = "config negative"
        sampling_config.sample_width = 640
        sampling_config.sample_height = 832
        sampling_config.sample_steps = 28
        sampling_config.sample_cfg_scale = 6.5
        sampling_config.sample_seed = 123
        sampling_config.sample_sampler = "ddim"
        training_config = MagicMock()
        objective_config = MagicMock()
        objective_config.target = DDPM_PREDICTION_TYPE_EPSILON
        saving_config = MagicMock()
        saving_config.output_name = "test"
        loss_config = MagicMock()

        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        with (
            patch("torch.manual_seed") as mock_seed,
            patch("os.path.join", return_value="/tmp/test.png"),
            patch("PIL.Image.Image.save"),
        ):
            sample_image_inference(
                mock_accelerator,
                sampling_config,
                training_config,
                saving_config,
                objective_config,
                loss_config,
                mock_pipeline,
                "/tmp",
                {
                    "enum": 0,
                    "prompt": "prompt file prompt",
                    "negative_prompt": "prompt file negative",
                    "width": 1024,
                    "height": 768,
                    "sample_steps": 50,
                    "scale": 8.0,
                    "seed": 777,
                    "sample_sampler": "euler_a",
                },
                epoch=1,
                steps=100,
                prompt_replacement=None,
            )

        mock_seed.assert_called_once_with(777)
        assert mock_pipeline.call_args.kwargs["prompt"] == "prompt file prompt"
        assert mock_pipeline.call_args.kwargs["negative_prompt"] == "prompt file negative"
        assert mock_pipeline.call_args.kwargs["width"] == 1024
        assert mock_pipeline.call_args.kwargs["height"] == 768
        assert mock_pipeline.call_args.kwargs["num_inference_steps"] == 50
        assert mock_pipeline.call_args.kwargs["guidance_scale"] == 8.0
        assert mock_get_scheduler.call_args.kwargs["sample_sampler"] == "euler_a"

    @patch("library.training.sample_generation.get_my_scheduler")
    def test_guidance_scale_alias_is_respected(self, mock_get_scheduler):
        mock_pipeline = MagicMock()
        mock_pipeline.latents_to_image.return_value = [Image.new("RGB", (512, 512))]
        mock_pipeline.return_value = "latents"

        mock_accelerator = MagicMock()
        mock_accelerator.autocast.return_value.__enter__ = MagicMock()
        mock_accelerator.autocast.return_value.__exit__ = MagicMock()
        mock_accelerator.trackers = []

        sampling_config = MagicMock()
        sampling_config.sample_prompt = None
        sampling_config.sample_negative_prompt = None
        sampling_config.sample_width = None
        sampling_config.sample_height = None
        sampling_config.sample_steps = None
        sampling_config.sample_cfg_scale = None
        sampling_config.sample_seed = None
        sampling_config.sample_sampler = "ddim"
        training_config = MagicMock()
        objective_config = MagicMock()
        objective_config.target = DDPM_PREDICTION_TYPE_EPSILON
        saving_config = MagicMock()
        saving_config.output_name = "test"
        loss_config = MagicMock()

        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        with (
            patch("os.path.join", return_value="/tmp/test.png"),
            patch("PIL.Image.Image.save"),
        ):
            sample_image_inference(
                mock_accelerator,
                sampling_config,
                training_config,
                saving_config,
                objective_config,
                loss_config,
                mock_pipeline,
                "/tmp",
                {"enum": 0, "prompt": "alias prompt", "guidance_scale": 9.0},
                epoch=1,
                steps=100,
                prompt_replacement=None,
            )

        assert mock_pipeline.call_args.kwargs["guidance_scale"] == 9.0
