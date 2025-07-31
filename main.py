import subprocess
import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# === CONFIGURATION ===
FRAMERATE = 21
WIDTH = 4032     # Use max resolution supported by Arducam
HEIGHT = 3040
NUM_CAMERAS = 6
output_dir = '/home/nvidia/imagestitching'
stitched_output_path = os.path.join(output_dir, "stitched_output.jpg")

os.makedirs(output_dir, exist_ok=True)

# === STEP 1: Capture from 4 Cameras Concurrently ===
print("\n📷 Starting capture from ArduCam cameras...")

cmd_template = (
    'gst-launch-1.0 nvarguscamerasrc sensor-id={id} num-buffers=1 ! '
    '"video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1" ! '
    'nvvidconv ! jpegenc ! filesink location={output}/capture{id}.jpg -e'
)

def capture_image(sensor_id):
    """Captures image from a specific camera."""
    cmd = cmd_template.format(
        id=sensor_id,
        width=WIDTH,
        height=HEIGHT,
        framerate=FRAMERATE,
        output=output_dir
    )
    print(f"→ Capturing from sensor-id={sensor_id}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Camera {sensor_id} failed to capture.")
    else:
        print(f"✔️ Captured image from sensor-id={sensor_id}")

with ThreadPoolExecutor(max_workers=NUM_CAMERAS) as executor:
    executor.map(capture_image, range(NUM_CAMERAS))

print("✅ All images captured.\n")

# === STEP 2: Verify and Load Images ===
image_paths = [f"{output_dir}/capture{i}.jpg" for i in range(NUM_CAMERAS)]
images = []
for path in image_paths:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"❌ Missing or empty image: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"❌ Failed to read image: {path}")
    print(f"✔️ Loaded {path} | shape: {img.shape}")
    images.append(img)

if len(images) < 2:
    raise ValueError("❌ Not enough valid images for stitching.")

# === STEP 3: Try OpenCV Stitcher First ===
print("\n🧵 Trying OpenCV Stitcher...")
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, stitched = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite(stitched_output_path, stitched)
    print(f"✅ [OpenCV] Panorama saved to: {stitched_output_path}")
else:
    print(f"⚠️ OpenCV stitcher failed. Status: {status}. Falling back to manual AKAZE stitching...")

    # === STEP 4: Manual AKAZE Stitching ===
    akaze = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    stitched = images[0]
    for i in range(1, len(images)):
        print(f"🔍 Matching image {i-1} and {i}...")

        kp1, des1 = akaze.detectAndCompute(stitched, None)
        kp2, des2 = akaze.detectAndCompute(images[i], None)

        if des1 is None or des2 is None or len(des1) < 1 or len(des2) < 1:
            print(f"⚠️ Skipping image {i}: not enough descriptors.")
            continue

        matches = matcher.knnMatch(des2, des1, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.9 * m[1].distance]

        if len(good) < 1:
            print(f"⚠️ Skipping image {i}: no good matches.")
            continue

        pts2 = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None:
            print(f"⚠️ Skipping image {i}: homography failed.")
            continue

        # Warp image[i] to stitched image's space
        h1, w1 = stitched.shape[:2]
        h2, w2 = images[i].shape[:2]
        canvas_width = w1 + w2
        canvas_height = max(h1, h2)
        warped = cv2.warpPerspective(images[i], H, (canvas_width, canvas_height))
        warped[0:h1, 0:w1] = stitched
        stitched = warped

    cv2.imwrite(stitched_output_path, stitched)
    print(f"✅ [AKAZE] Panorama saved to: {stitched_output_path}")

# === STEP 5: Show Preview ===
preview = cv2.resize(stitched, (1280, 720)) if stitched.shape[1] > 1280 else stitched
cv2.imshow("🖼️ Panorama Preview", preview)
print("👁️ Press any key to close the preview.")
cv2.waitKey(0)
cv2.destroyAllWindows()

