
import cv2
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor


# ==========================================
# CONFIGURATION
# ==========================================

TILE_SIZE = 32
THRESHOLD = 25
WORKERS = 4

# 3x3 cellular neighborhood
CELL_KERNEL = np.ones((3, 3), dtype=np.uint8)


# ==========================================
# STEP 1: SEQUENTIAL MOTION DETECTION
# ==========================================

def sequential_detection(previous, current):

    height, width = current.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):

            y2 = min(y + TILE_SIZE, height)
            x2 = min(x + TILE_SIZE, width)

            old_tile = previous[y:y2, x:x2]
            new_tile = current[y:y2, x:x2]

            difference = cv2.absdiff(old_tile, new_tile)

            _, mask = cv2.threshold(
                difference,
                THRESHOLD,
                255,
                cv2.THRESH_BINARY
            )

            result[y:y2, x:x2] = mask

    return result


# ==========================================
# STEP 2: PARALLEL CELL PROCESSING
# ==========================================

def process_tile(arguments):

    previous, current, x, y, x2, y2 = arguments

    old_tile = previous[y:y2, x:x2]
    new_tile = current[y:y2, x:x2]

    difference = cv2.absdiff(old_tile, new_tile)

    _, mask = cv2.threshold(
        difference,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    return x, y, x2, y2, mask


def parallel_detection(previous, current, executor):

    height, width = current.shape

    tasks = []

    # Divide image into smaller tiles
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):

            y2 = min(y + TILE_SIZE, height)
            x2 = min(x + TILE_SIZE, width)

            tasks.append(
                (previous, current, x, y, x2, y2)
            )

    # Process tiles concurrently
    results = executor.map(process_tile, tasks)

    output = np.zeros((height, width), dtype=np.uint8)

    # Combine the processed tiles
    for x, y, x2, y2, mask in results:

        output[y:y2, x:x2] = mask

    return output


# ==========================================
# STEP 3: CELLULAR NEIGHBORHOOD RULES
# ==========================================

def cellular_refinement(mask):

    # Count active pixels in each 3x3 neighborhood
    neighbor_count = cv2.filter2D(
        mask // 255,
        cv2.CV_16U,
        CELL_KERNEL,
        borderType=cv2.BORDER_CONSTANT
    )

    # Cellular rule:
    # A pixel remains active if at least 5
    # of the 9 neighborhood positions are active.

    refined = np.where(
        neighbor_count >= 5,
        255,
        0
    ).astype(np.uint8)

    return refined


# ==========================================
# STEP 4: IDENTIFY MOVING OBJECTS
# ==========================================

def draw_motion_boxes(frame, mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny motion regions
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Motion Detected",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        motion_count += 1

    return frame, motion_count


# ==========================================
# STEP 5: MAIN PROGRAM
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video file path"
    )

    args = parser.parse_args()

    # Select webcam or video file
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    # Parallel worker pool
    executor = ThreadPoolExecutor(max_workers=WORKERS)

    previous_frame = None

    sequential_times = []
    parallel_times = []

    frame_number = 0

    print("Motion detection started.")
    print("Press Q to quit.")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Resize input frame
            frame = cv2.resize(frame, (640, 480))

            # Convert to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Reduce small image noise
            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            if previous_frame is None:
                previous_frame = gray
                continue

            # -------------------------------
            # SEQUENTIAL PROCESSING
            # -------------------------------

            start = time.perf_counter()

            sequential_mask = sequential_detection(
                previous_frame,
                gray
            )

            sequential_mask = cellular_refinement(
                sequential_mask
            )

            sequential_time = (
                time.perf_counter() - start
            ) * 1000

            # -------------------------------
            # PARALLEL PROCESSING
            # -------------------------------

            start = time.perf_counter()

            parallel_mask = parallel_detection(
                previous_frame,
                gray,
                executor
            )

            parallel_mask = cellular_refinement(
                parallel_mask
            )

            parallel_time = (
                time.perf_counter() - start
            ) * 1000

            sequential_times.append(sequential_time)
            parallel_times.append(parallel_time)

            frame_number += 1

            # -------------------------------
            # DRAW DETECTED OBJECTS
            # -------------------------------

            output_frame = frame.copy()

            output_frame, motion_count = draw_motion_boxes(
                output_frame,
                parallel_mask
            )

            # Display performance information
            cv2.putText(
                output_frame,
                f"Sequential: {sequential_time:.2f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                f"Parallel: {parallel_time:.2f} ms",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Moving regions: {motion_count}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------------------------------
            # DISPLAY RESULTS
            # -------------------------------

            cv2.imshow(
                "Motion Detection - Parallel Cellular",
                output_frame
            )

            cv2.imshow(
                "Motion Mask",
                parallel_mask
            )

            previous_frame = gray.copy()

            # Quit when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        cap.release()
        executor.shutdown(wait=True)
        cv2.destroyAllWindows()

    # ======================================
    # STEP 6: PERFORMANCE COMPARISON
    # ======================================

    if sequential_times:

        avg_sequential = np.mean(sequential_times)
        avg_parallel = np.mean(parallel_times)

        speedup = avg_sequential / avg_parallel

        print("\n===== PERFORMANCE RESULTS =====")

        print(
            f"Frames processed: {frame_number}"
        )

        print(
            f"Sequential average: {avg_sequential:.2f} ms"
        )

        print(
            f"Parallel average: {avg_parallel:.2f} ms"
        )

        print(
            f"Sequential FPS: {1000 / avg_sequential:.2f}"
        )

        print(
            f"Parallel FPS: {1000 / avg_parallel:.2f}"
        )

        print(
            f"Speedup: {speedup:.2f}x"
        )

        # Performance graph
        plt.figure(figsize=(9, 5))

        plt.plot(
            sequential_times,
            label="Sequential"
        )

        plt.plot(
            parallel_times,
            label="Parallel Cellular"
        )

        plt.xlabel("Frame Number")
        plt.ylabel("Processing Time (ms)")

        plt.title("Motion Detection Performance")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
