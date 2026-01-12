import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from library.strategies.sdxl.caching import SdxlLatentsPipelineStrategy


def compare_interpolations():
    image_path = "/tests/assets/ce01fa2f12d867e209c36516d65df150.png"

    print(f"Loading {image_path}...")
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        original_image = Image.open(image_path)
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Simulate bucketing logic: Fit within 2048x2048 while keeping AR
    MAX_SIZE = 2048
    w, h = original_image.size
    scale = min(MAX_SIZE / w, MAX_SIZE / h)

    # Only downscale if larger (bucketing wouldn't upscale just to fit max, usually)
    # But for this test, let's force the scale to see the effect
    # If the image is smaller than 2048, this scale > 1 (upscaling)
    # If the image is larger, scale < 1 (downscaling)

    new_w = int(w * scale)
    new_h = int(h * scale)

    TARGET_SIZE = (new_w, new_h)

    print(f"Original: {w}x{h}, Target (AR preserved): {new_w}x{new_h}, Scale: {scale:.4f}")

    methods = [("area", "cv2.INTER_AREA"), ("hamming", "PIL.HAMMING (Default)"), ("bicubic", "PIL.BICUBIC"), ("lanczos", "PIL.LANCZOS")]

    strategy = SdxlLatentsPipelineStrategy()

    results = []

    # We pass BOTH target_size and resized_size as the same because we are testing PURE RESIZE here, not crop
    # preprocess_image logic:
    #   if resized_size is None -> resize to target_size
    #   if resized_size provided -> resize to resized_size, then crop to target_size
    # Here we want result = target_size, so we just set both equal.

    for method_name, display_name in methods:
        print(f"Processing: {display_name}...")

        tensor = strategy.preprocess_image(
            original_image, target_size=TARGET_SIZE, resized_size=TARGET_SIZE, resize_interpolation=method_name
        )

        # Convert back to PIL for viewing
        arr = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(arr)

        # Add label
        draw = ImageDraw.Draw(img)
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw text with outline for visibility
        text = display_name
        text_pos = (10, 10)
        # Helper for outline
        x, y = text_pos
        shadow_color = "black"
        fill_color = "white"

        # Draw outline/shadow
        draw.text((x - 1, y - 1), text, font=font, fill=shadow_color)
        draw.text((x + 1, y - 1), text, font=font, fill=shadow_color)
        draw.text((x - 1, y + 1), text, font=font, fill=shadow_color)
        draw.text((x + 1, y + 1), text, font=font, fill=shadow_color)
        draw.text(text_pos, text, font=font, fill=fill_color)

        results.append(img)
        img.save(f"debug_interp_{method_name}.jpg", quality=100)

    # Create a grid/strip
    w, h = results[0].size
    grid_img = Image.new("RGB", (w * len(results), h))
    for i, img in enumerate(results):
        grid_img.paste(img, (i * w, 0))

    grid_img.save("debug_interp_comparison.jpg", quality=100)
    print(f"\nSaved individual files (debug_interp_*.jpg) and combined grid: {os.path.abspath('../debug_interp_comparison.jpg')}")


if __name__ == "__main__":
    compare_interpolations()
