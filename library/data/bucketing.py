import math


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    """
    Generate bucket resolutions for training with aspect ratio bucketing.

    This is the same algorithm used in the _deprecated BucketManager.

    Args:
        max_reso: Tuple of (max_width, max_height) defining the maximum resolution.
        min_size: Minimum dimension size.
        max_size: Maximum dimension size.
        divisible: All dimensions must be divisible by this.

    Returns:
        Sorted list of (width, height) tuples representing bucket resolutions.
    """
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    # Add square bucket
    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    # Generate aspect ratio buckets
    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    resos_list = list(resos)
    resos_list.sort()
    return resos_list


def select_bucket(
    image_width: int,
    image_height: int,
    bucket_resos: list[tuple[int, int]],
    no_upscale: bool = False,
    max_area: int | None = None,
    reso_steps: int = 64,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select the best bucket for an image.

    This is ported from BucketManager.select_bucket() for compatibility.

    Args:
        image_width: Original image width.
        image_height: Original image height.
        bucket_resos: List of available bucket resolutions.
        no_upscale: If True, never upscale images, only shrink.
        max_area: Maximum pixel area (for no_upscale mode).
        reso_steps: Resolution step size for rounding.

    Returns:
        Tuple of (bucket_reso, resized_size).
    """
    aspect_ratio = image_width / image_height

    if not no_upscale:
        # Standard mode: match to predefined bucket by aspect ratio
        bucket_resos_set = set(bucket_resos)
        reso = (image_width, image_height)

        if reso not in bucket_resos_set:
            # Find bucket with closest aspect ratio
            bucket_ars = [(w / h, (w, h)) for w, h in bucket_resos]
            best_match = min(bucket_ars, key=lambda x: abs(x[0] - aspect_ratio))
            reso = best_match[1]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            # Image is wider, match height
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
    else:
        # No upscale mode: only shrink, create custom bucket
        if max_area is None:
            max_area = max(w * h for w, h in bucket_resos) if bucket_resos else 1024 * 1024

        if image_width * image_height > max_area:
            # Shrink maintaining aspect ratio
            resized_width = math.sqrt(max_area * aspect_ratio)
            resized_height = max_area / resized_width

            # Round to reso_steps
            def round_to_steps(x):
                x = int(x + 0.5)
                return x - x % reso_steps

            b_width_rounded = round_to_steps(resized_width)
            b_height_in_wr = round_to_steps(b_width_rounded / aspect_ratio)
            ar_width_rounded = b_width_rounded / b_height_in_wr if b_height_in_wr else 0

            b_height_rounded = round_to_steps(resized_height)
            b_width_in_hr = round_to_steps(b_height_rounded * aspect_ratio)
            ar_height_rounded = b_width_in_hr / b_height_rounded if b_height_rounded else 0

            if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
            else:
                resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
        else:
            resized_size = (image_width, image_height)

        # Ensure minimal dimensions to avoid 0x0 buckets for small images
        if resized_size[0] < reso_steps or resized_size[1] < reso_steps:
            resized_size = (
                max(resized_size[0], reso_steps),
                max(resized_size[1], reso_steps),
            )

        # Bucket size is resized size rounded down to reso_steps
        bucket_width = resized_size[0] - resized_size[0] % reso_steps
        bucket_height = resized_size[1] - resized_size[1] % reso_steps
        reso = (bucket_width, bucket_height)

    return reso, resized_size
