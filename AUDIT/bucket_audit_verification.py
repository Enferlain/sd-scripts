
import math
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- Legacy Implementation (copied from library/data/_deprecated/data_structures.py) ---

class LegacyBucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        # self.buckets = []  # Not needed for this test

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            # self.buckets.append([])

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def make_bucket_resolutions(self):
         # Helper to generate resolutions for testing parity
        max_width, max_height = self.max_reso
        max_area = max_width * max_height

        resos = set()

        # Add square bucket
        width = int(math.sqrt(max_area) // self.reso_steps) * self.reso_steps
        resos.add((width, width))

        # Generate aspect ratio buckets
        width = self.min_size
        while width <= self.max_size:
            height = min(self.max_size, int((max_area // width) // self.reso_steps) * self.reso_steps)
            if height >= self.min_size:
                resos.add((width, height))
                resos.add((height, width))
            width += self.reso_steps

        resos = list(resos)
        resos.sort()
        return resos

    def set_predefined_resos(self, resos):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width

                # Check aspect ratio assumption
                # assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr if b_height_in_wr > 0 else 0

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded if b_height_rounded > 0 else 0

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
            else:
                resized_size = (image_width, image_height)

            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)
        return reso, resized_size


# --- New Implementation Import ---
try:
    from library.data import make_bucket_resolutions, select_bucket as new_select_bucket
except ImportError as e:
    print(f"Could not import new implementation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def run_comparison():
    logger.info("Starting Bucket Logic Audit...")

    # Common Parameters
    MAX_RESO = (1024, 1024)
    MIN_SIZE = 256
    MAX_SIZE = 1024
    RESO_STEPS = 64

    # Test Cases: (width, height)
    test_cases = [
        (1024, 1024), # Exact match
        (512, 512),   # Smaller square
        (2000, 2000), # Larger square
        (768, 1024),  # Portrait
        (1024, 768),  # Landscape
        (100, 100),   # Small image (< min_size)
        (30, 30),     # Tiny image (< step)
        (1000, 500),  # 2:1
        (500, 1000),  # 1:2
        (1234, 567),  # Random sizes
        (1920, 1080), # HD
        (4096, 4096), # Huge
    ]

    # --- Test 1: no_upscale = False ---
    logger.info("\n=== Test Case 1: Standard Mode (no_upscale=False) ===")

    legacy_mgr = LegacyBucketManager(False, MAX_RESO, MIN_SIZE, MAX_SIZE, RESO_STEPS)
    legacy_resos = legacy_mgr.make_bucket_resolutions()
    legacy_mgr.set_predefined_resos(legacy_resos)

    # New impl bucket resolution generation
    new_resos = make_bucket_resolutions(MAX_RESO, MIN_SIZE, MAX_SIZE, RESO_STEPS)

    # Verify bucket lists match
    if set(legacy_resos) != set(new_resos):
        logger.error("MISMATCH in bucket resolution generation!")
        logger.error(f"Legacy count: {len(legacy_resos)}, New count: {len(new_resos)}")
    else:
        logger.info("Bucket resolution lists match.")

    for w, h in test_cases:
        l_reso, l_resize = legacy_mgr.select_bucket(w, h)

        # New function signature:
        # select_bucket(image_width, image_height, bucket_resos, no_upscale, max_area, reso_steps)
        max_area = MAX_RESO[0] * MAX_RESO[1]
        n_reso, n_resize = new_select_bucket(w, h, new_resos, False, max_area, RESO_STEPS)

        match = (l_reso == n_reso) and (l_resize == n_resize)
        status = "PASS" if match else "FAIL"

        if not match:
             logger.warning(f"[{status}] Input: {w}x{h}")
             logger.warning(f"  Legacy: Bucket={l_reso}, Resize={l_resize}")
             logger.warning(f"  New:    Bucket={n_reso}, Resize={n_resize}")


    # --- Test 2: no_upscale = True ---
    logger.info("\n=== Test Case 2: No Upscale Mode (no_upscale=True) ===")

    legacy_mgr_nu = LegacyBucketManager(True, MAX_RESO, MIN_SIZE, MAX_SIZE, RESO_STEPS)
    # Note: Legacy select_bucket with no_upscale doesn't use predefined_resos

    for w, h in test_cases:
        # Legacy
        l_reso, l_resize = legacy_mgr_nu.select_bucket(w, h)

        # New
        n_reso, n_resize = new_select_bucket(w, h, new_resos, True, max_area, RESO_STEPS)

        match = (l_reso == n_reso) and (l_resize == n_resize)
        status = "PASS" if match else "FAIL"

        if not match:
             logger.warning(f"[{status}] Input: {w}x{h}")
             logger.warning(f"  Legacy: Bucket={l_reso}, Resize={l_resize}")
             logger.warning(f"  New:    Bucket={n_reso}, Resize={n_resize}")

    # --- Test 3: The Zero-Dimension Edge Case ---
    logger.info("\n=== Test Case 3: Zero Dimension Edge Case ===")
    tiny_w, tiny_h = 30, 30

    # Legacy
    l_reso, l_resize = legacy_mgr_nu.select_bucket(tiny_w, tiny_h)

    # New
    n_reso, n_resize = new_select_bucket(tiny_w, tiny_h, new_resos, True, max_area, RESO_STEPS)

    logger.info(f"Input: {tiny_w}x{tiny_h} (step={RESO_STEPS})")
    logger.info(f"Legacy Result: Bucket={l_reso}, Resize={l_resize}")
    logger.info(f"New Result:    Bucket={n_reso}, Resize={n_resize}")

    if l_reso[0] == 0 or l_reso[1] == 0:
        logger.info("Legacy behavior confirms 0-dimension bucket generation.")

    if l_reso == n_reso:
        logger.info("New implementation faithfully reproduces this edge case.")
    else:
        logger.warning("New implementation handles this differently.")

if __name__ == "__main__":
    run_comparison()
