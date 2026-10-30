# Recovered FSOC benchmark source cells
# Extracted read-only from Virtual_Camera_Tracking.ipynb
# NOT EXECUTED. Inspect dependencies before running.



# ======================================================================
# ORIGINAL NOTEBOOK CELL 331
# ======================================================================
# ============================================================
# STEP 53F: MULTI-TRAJECTORY CLOSED-LOOP BENCHMARK
# ============================================================
#
# 30 Hz Control
# 10 Hz YOLO
# Kalman prediction between detections
# Predictive pan/tilt control
# Single YOLO inference per vision frame
# ============================================================

import time
import numpy as np
import cv2

# ------------------------------------------------------------
# Benchmark configuration
# ------------------------------------------------------------

BENCHMARK_DURATION = 20.0

FPS_CONTROL = 30
DT_CONTROL = 1.0 / FPS_CONTROL

YOLO_FPS = 10
YOLO_INTERVAL = 3

TRAJECTORIES = [
    "straight",
    "circular",
    "figure8",
    "spiral",
    "random"
]

TARGET_SPEED = 40.0

LOCK_THRESHOLD = 10.0

MEASUREMENT_NOISE_STD = 2.0

KP_ANGLE = 6.0
KV_ANGLE = 1.0

RANGE = 1000.0


# ------------------------------------------------------------
# Single YOLO inference helper
# ------------------------------------------------------------

def yolo_detect_once(model, frame, confidence_threshold=0.25):

    results = model(
        frame,
        verbose=False
    )

    if len(results) == 0:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes

    best_index = int(
        np.argmax(
            boxes.conf.cpu().numpy()
        )
    )

    confidence = float(
        boxes.conf[best_index].cpu().item()
    )

    if confidence < confidence_threshold:
        return None

    xyxy = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = xyxy

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return (
        float(cx),
        float(cy),
        confidence
    )


# ------------------------------------------------------------
# World velocity -> angular velocity
# ------------------------------------------------------------

def velocity_to_angular(
    target_x,
    target_y,
    vx,
    vy,
    camera,
    range_value=1000.0
):

    dx = target_x - camera.x
    dy = target_y - camera.y

    pan_den = (
        range_value**2
        + dx**2
    )

    tilt_den = (
        range_value**2
        + dy**2
    )

    angular_vx = (
        range_value
        * vx
        / pan_den
    ) * (180.0 / np.pi)

    angular_vy = (
        range_value
        * vy
        / tilt_den
    ) * (180.0 / np.pi)

    return angular_vx, angular_vy


# ------------------------------------------------------------
# Run one trajectory benchmark
# ------------------------------------------------------------

def run_trajectory_benchmark(
    motion_type,
    speed=40.0
):

    total_frames = int(
        BENCHMARK_DURATION
        * FPS_CONTROL
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = AngularCamera()

    camera.width = 640
    camera.height = 480

    camera.hfov = 4.0
    camera.vfov = 3.0

    camera.max_pan_speed = 5.0
    camera.max_tilt_speed = 5.0

    camera.x = 1000.0
    camera.y = 1000.0

    camera.pan = 0.0
    camera.tilt = 0.0


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = BoundedTarget(
        motion_type=motion_type,
        center=(1000.0, 1000.0),
        amplitude_x=500.0,
        amplitude_y=350.0,
        speed=speed
    )


    # --------------------------------------------------------
    # Kalman filter
    # --------------------------------------------------------

    kf = create_kalman()

    kf_initialized = False


    # --------------------------------------------------------
    # Runtime statistics
    # --------------------------------------------------------

    errors = []

    detection_errors = []

    detection_count = 0
    vision_frames = 0

    prediction_frames = 0

    lock_count = 0

    miss_count = 0

    current_miss = 0
    max_miss = 0

    visible_frames = 0

    start_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for frame_id in range(total_frames):

        # ----------------------------------------------------
        # Update target
        # ----------------------------------------------------

        target_x, target_y = target.update(
            DT_CONTROL
        )


        # ----------------------------------------------------
        # Render FPA
        # ----------------------------------------------------

        frame, true_px, true_py = (
            render_consistent_fpa(
                target_x,
                target_y,
                camera,
                beacon_size=10
            )
        )

        visible = beacon_visible(
            true_px,
            true_py,
            camera
        )

        if visible:
            visible_frames += 1


        detection = None


        # ====================================================
        # YOLO @ 10 Hz
        # ====================================================

        if frame_id % YOLO_INTERVAL == 0:

            vision_frames += 1

            if visible:

                detection = yolo_detect_once(
                    yolo_model,
                    frame
                )

            # ------------------------------------------------
            # Detection available
            # ------------------------------------------------

            if detection is not None:

                detection_count += 1

                current_miss = 0

                det_x, det_y, confidence = detection


                # --------------------------------------------
                # Detection centroid error
                # --------------------------------------------

                det_error = np.sqrt(
                    (det_x - true_px)**2
                    +
                    (det_y - true_py)**2
                )

                detection_errors.append(
                    det_error
                )


                # --------------------------------------------
                # FPA -> world measurement
                # --------------------------------------------

                angular_error_x = (
                    (det_x - camera.width / 2)
                    * camera.hfov
                    / camera.width
                )

                angular_error_y = (
                    (det_y - camera.height / 2)
                    * camera.vfov
                    / camera.height
                )

                measured_x = (
                    camera.x
                    +
                    np.tan(
                        np.radians(
                            camera.pan
                            + angular_error_x
                        )
                    )
                    * RANGE
                )

                measured_y = (
                    camera.y
                    +
                    np.tan(
                        np.radians(
                            camera.tilt
                            + angular_error_y
                        )
                    )
                    * RANGE
                )


                # --------------------------------------------
                # Measurement noise
                # --------------------------------------------

                measured_x += np.random.normal(
                    0,
                    MEASUREMENT_NOISE_STD
                )

                measured_y += np.random.normal(
                    0,
                    MEASUREMENT_NOISE_STD
                )


                # --------------------------------------------
                # First detection initializes Kalman
                # --------------------------------------------

                if not kf_initialized:

                    kf.x = np.array([
                        [measured_x],
                        [measured_y],
                        [0.0],
                        [0.0]
                    ])

                    kf_initialized = True

                else:

                    kf.predict()

                    kf.update([
                        measured_x,
                        measured_y
                    ])


            # ------------------------------------------------
            # Detection missed
            # ------------------------------------------------

            else:

                miss_count += 1
                current_miss += 1

                max_miss = max(
                    max_miss,
                    current_miss
                )

                if kf_initialized:
                    kf.predict()


        # ====================================================
        # KALMAN PREDICTION BETWEEN YOLO FRAMES
        # ====================================================

        else:

            if kf_initialized:

                kf.predict()

                prediction_frames += 1


        # ====================================================
        # CAMERA CONTROL
        # ====================================================

        if kf_initialized:

            estimated_x = float(
                kf.x[0, 0]
            )

            estimated_y = float(
                kf.x[1, 0]
            )

            estimated_vx = float(
                kf.x[2, 0]
            )

            estimated_vy = float(
                kf.x[3, 0]
            )


            # ----------------------------------------------
            # Predicted target position on FPA
            # ----------------------------------------------

            pred_px, pred_py = world_to_fpa(
                estimated_x,
                estimated_y,
                camera
            )


            error_x = (
                pred_px
                - camera.width / 2
            )

            error_y = (
                pred_py
                - camera.height / 2
            )


            # ----------------------------------------------
            # Pixel -> angular error
            # ----------------------------------------------

            angular_error_x = (
                error_x
                * camera.hfov
                / camera.width
            )

            angular_error_y = (
                error_y
                * camera.vfov
                / camera.height
            )


            # ----------------------------------------------
            # Velocity feed-forward
            # ----------------------------------------------

            angular_vx, angular_vy = (
                velocity_to_angular(
                    estimated_x,
                    estimated_y,
                    estimated_vx,
                    estimated_vy,
                    camera,
                    RANGE
                )
            )


            # ----------------------------------------------
            # Predictive controller
            # ----------------------------------------------

            pan_rate = (
                KP_ANGLE
                * angular_error_x
                +
                KV_ANGLE
                * angular_vx
            )

            tilt_rate = (
                KP_ANGLE
                * angular_error_y
                +
                KV_ANGLE
                * angular_vy
            )


            # ----------------------------------------------
            # Physical camera limits
            # ----------------------------------------------

            pan_rate = np.clip(
                pan_rate,
                -camera.max_pan_speed,
                camera.max_pan_speed
            )

            tilt_rate = np.clip(
                tilt_rate,
                -camera.max_tilt_speed,
                camera.max_tilt_speed
            )


        else:

            # ----------------------------------------------
            # No estimate yet: raster acquisition
            # ----------------------------------------------

            search = RasterSearchController(
                pan_range=(-8.0, 8.0),
                tilt_range=(-6.0, 6.0),
                pan_step=2.0,
                tilt_step=1.5,
                speed=5.0
            )

            search.reset(camera)

            pan_rate, tilt_rate = (
                search.update(
                    camera,
                    DT_CONTROL
                )
            )


        # ====================================================
        # MOVE CAMERA
        # ====================================================

        camera.move(
            pan_rate,
            tilt_rate,
            DT_CONTROL
        )


        # ====================================================
        # TRUE ERROR
        # ====================================================

        actual_px, actual_py = world_to_fpa(
            target_x,
            target_y,
            camera
        )

        error = np.sqrt(
            (actual_px - camera.width / 2)**2
            +
            (actual_py - camera.height / 2)**2
        )

        errors.append(error)


        if error <= LOCK_THRESHOLD:
            lock_count += 1


    # ========================================================
    # RESULTS
    # ========================================================

    elapsed = time.time() - start_time

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    final_error = errors[-1]

    lock_percentage = (
        lock_count
        / total_frames
        * 100
    )

    detection_percentage = (
        detection_count
        / max(1, vision_frames)
        * 100
    )

    visible_detection_percentage = (
        detection_count
        / max(1, visible_frames / YOLO_INTERVAL)
        * 100
    )

    avg_detection_error = (
        np.mean(detection_errors)
        if detection_errors
        else np.nan
    )

    runtime_fps = (
        total_frames / elapsed
    )

    return {
        "Trajectory": motion_type,
        "Final Error (px)": final_error,
        "Average Error (px)": avg_error,
        "Maximum Error (px)": max_error,
        "Lock <=10px (%)": lock_percentage,
        "YOLO Detection (%)": detection_percentage,
        "Avg Detection Error (px)": avg_detection_error,
        "Visible Frames": visible_frames,
        "YOLO Frames": vision_frames,
        "Max Consecutive Miss": max_miss,
        "Runtime FPS": runtime_fps
    }


# ============================================================
# RUN BENCHMARK
# ============================================================

print("=" * 70)
print("STEP 53F: MULTI-TRAJECTORY CLOSED-LOOP BENCHMARK")
print("=" * 70)

results_53f = []

for motion in TRAJECTORIES:

    print()
    print(f"Running: {motion.upper()} ...")

    result = run_trajectory_benchmark(
        motion_type=motion,
        speed=TARGET_SPEED
    )

    results_53f.append(result)

    print(
        f"  Final Error : "
        f"{result['Final Error (px)']:.2f} px"
    )

    print(
        f"  Average     : "
        f"{result['Average Error (px)']:.2f} px"
    )

    print(
        f"  Max Error   : "
        f"{result['Maximum Error (px)']:.2f} px"
    )

    print(
        f"  Lock <=10px : "
        f"{result['Lock <=10px (%)']:.2f}%"
    )

    print(
        f"  YOLO Detect : "
        f"{result['YOLO Detection (%)']:.2f}%"
    )

    print(
        f"  Runtime FPS : "
        f"{result['Runtime FPS']:.2f}"
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

results_53f_df = pd.DataFrame(
    results_53f
)

print()
print("=" * 70)
print("STEP 53F SUMMARY")
print("=" * 70)

display(
    results_53f_df.round(2)
)

print("=" * 70)


# ======================================================================
# ORIGINAL NOTEBOOK CELL 332
# ======================================================================
# ============================================================
# STEP 53G: FAIR TRAJECTORY BENCHMARK CONFIGURATION
# ============================================================

# The previous benchmark used amplitudes of 500 x 350 px.
# With a 4° x 3° FOV and ~1000 px virtual range, this puts
# much of the trajectory far outside the initial camera FOV.
#
# This benchmark starts the beacon near the optical axis so
# that we can evaluate TRACKING independently from ACQUISITION.

FAIR_CENTER = (1005.0, 1000.0)

FAIR_AMPLITUDE_X = 25.0
FAIR_AMPLITUDE_Y = 18.0

FAIR_TARGET_SPEED = 40.0

print("=" * 70)
print("STEP 53G: FAIR TRAJECTORY BENCHMARK CONFIGURATION")
print("=" * 70)

print(f"Initial target : {FAIR_CENTER}")
print(f"Amplitude X    : ±{FAIR_AMPLITUDE_X} px")
print(f"Amplitude Y    : ±{FAIR_AMPLITUDE_Y} px")
print(f"Target speed   : {FAIR_TARGET_SPEED} px/s")
print()
print("Purpose:")
print("  ✓ Evaluate tracking independently from acquisition")
print("  ✓ Keep initial beacon inside the narrow FOV")
print("  ✓ Prevent false benchmark failures caused by visibility")
print("=" * 70)


# ======================================================================
# ORIGINAL NOTEBOOK CELL 333
# ======================================================================
# ============================================================
# STEP 53H: CORRECTED MULTI-TRAJECTORY TRACKING BENCHMARK
# ============================================================

import time
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TEST_DURATION = 20.0

FPS_CONTROL = 30
DT_CONTROL = 1.0 / FPS_CONTROL

YOLO_FPS = 10
YOLO_INTERVAL = 3

LOCK_THRESHOLD = 10.0

MEASUREMENT_NOISE_STD = 2.0

KP_ANGLE = 6.0
KV_ANGLE = 1.0

RANGE = 1000.0

TRAJECTORIES_53H = [
    "straight",
    "circular",
    "figure8",
    "spiral",
    "random"
]

TARGET_SPEED = FAIR_TARGET_SPEED

# ------------------------------------------------------------
# YOLO single-inference function
# ------------------------------------------------------------

def detect_yolo_once_53h(frame):

    results = yolo_model(
        frame,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    confidence = float(
        confidences[best_index]
    )

    if confidence < 0.25:
        return None

    box = (
        result.boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = box

    cx = (
        float(x1) + float(x2)
    ) / 2.0

    cy = (
        float(y1) + float(y2)
    ) / 2.0

    return (
        cx,
        cy,
        confidence
    )


# ------------------------------------------------------------
# Velocity conversion
# ------------------------------------------------------------

def angular_velocity_53h(
    target_x,
    target_y,
    vx,
    vy,
    camera
):

    dx = target_x - camera.x
    dy = target_y - camera.y

    pan_den = (
        RANGE**2
        + dx**2
    )

    tilt_den = (
        RANGE**2
        + dy**2
    )

    vx_angle = (
        RANGE * vx / pan_den
    ) * (180.0 / np.pi)

    vy_angle = (
        RANGE * vy / tilt_den
    ) * (180.0 / np.pi)

    return vx_angle, vy_angle


# ------------------------------------------------------------
# Single trajectory benchmark
# ------------------------------------------------------------

def benchmark_trajectory_53h(
    motion_type
):

    total_frames = int(
        TEST_DURATION
        * FPS_CONTROL
    )

    # ========================================================
    # CAMERA
    # ========================================================

    camera = AngularCamera()

    camera.width = 640
    camera.height = 480

    camera.hfov = 4.0
    camera.vfov = 3.0

    camera.max_pan_speed = 5.0
    camera.max_tilt_speed = 5.0

    camera.x = 1000.0
    camera.y = 1000.0

    camera.pan = 0.0
    camera.tilt = 0.0


    # ========================================================
    # TARGET
    # ========================================================

    target = BoundedTarget(
        motion_type=motion_type,
        center=FAIR_CENTER,
        amplitude_x=FAIR_AMPLITUDE_X,
        amplitude_y=FAIR_AMPLITUDE_Y,
        speed=TARGET_SPEED
    )


    # ========================================================
    # KALMAN
    # ========================================================

    kf = create_kalman()

    kalman_initialized = False


    # ========================================================
    # PERSISTENT SEARCH CONTROLLER
    # ========================================================

    search = RasterSearchController(
        pan_range=(-8.0, 8.0),
        tilt_range=(-6.0, 6.0),
        pan_step=2.0,
        tilt_step=1.5,
        speed=5.0
    )

    search.reset(camera)


    # ========================================================
    # METRICS
    # ========================================================

    errors = []

    detection_errors = []

    detection_count = 0
    vision_frames = 0

    visible_frames = 0

    prediction_frames = 0

    lock_count = 0

    miss_count = 0

    current_miss = 0
    max_miss = 0

    start_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for frame_id in range(total_frames):

        # ----------------------------------------------------
        # TARGET UPDATE
        # ----------------------------------------------------

        target_x, target_y = target.update(
            DT_CONTROL
        )


        # ----------------------------------------------------
        # RENDER FPA
        # ----------------------------------------------------

        frame, true_px, true_py = (
            render_consistent_fpa(
                target_x,
                target_y,
                camera,
                beacon_size=10
            )
        )

        visible = beacon_visible(
            true_px,
            true_py,
            camera
        )

        if visible:
            visible_frames += 1


        detection = None


        # ====================================================
        # YOLO — 10 Hz
        # ====================================================

        if frame_id % YOLO_INTERVAL == 0:

            vision_frames += 1

            if visible:

                detection = detect_yolo_once_53h(
                    frame
                )


            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            if detection is not None:

                detection_count += 1

                current_miss = 0

                det_x, det_y, confidence = (
                    detection
                )


                # --------------------------------------------
                # Pixel detection accuracy
                # --------------------------------------------

                det_error = np.sqrt(
                    (det_x - true_px)**2
                    +
                    (det_y - true_py)**2
                )

                detection_errors.append(
                    det_error
                )


                # --------------------------------------------
                # FPA -> world coordinates
                # --------------------------------------------

                angular_error_x = (
                    (det_x - camera.width / 2)
                    * camera.hfov
                    / camera.width
                )

                angular_error_y = (
                    (det_y - camera.height / 2)
                    * camera.vfov
                    / camera.height
                )


                measured_x = (
                    camera.x
                    +
                    np.tan(
                        np.radians(
                            camera.pan
                            + angular_error_x
                        )
                    )
                    * RANGE
                )

                measured_y = (
                    camera.y
                    +
                    np.tan(
                        np.radians(
                            camera.tilt
                            + angular_error_y
                        )
                    )
                    * RANGE
                )


                # --------------------------------------------
                # Measurement noise
                # --------------------------------------------

                measured_x += np.random.normal(
                    0,
                    MEASUREMENT_NOISE_STD
                )

                measured_y += np.random.normal(
                    0,
                    MEASUREMENT_NOISE_STD
                )


                # --------------------------------------------
                # Kalman initialization
                # --------------------------------------------

                if not kalman_initialized:

                    kf.x = np.array([
                        [measured_x],
                        [measured_y],
                        [0.0],
                        [0.0]
                    ])

                    kalman_initialized = True

                else:

                    kf.predict()

                    kf.update([
                        measured_x,
                        measured_y
                    ])


            # ------------------------------------------------
            # MISS
            # ------------------------------------------------

            else:

                miss_count += 1

                current_miss += 1

                max_miss = max(
                    max_miss,
                    current_miss
                )

                if kalman_initialized:

                    kf.predict()


        # ====================================================
        # BETWEEN YOLO FRAMES
        # ====================================================

        else:

            if kalman_initialized:

                kf.predict()

                prediction_frames += 1


        # ====================================================
        # CONTROL
        # ====================================================

        if kalman_initialized:

            estimated_x = float(
                kf.x[0, 0]
            )

            estimated_y = float(
                kf.x[1, 0]
            )

            estimated_vx = float(
                kf.x[2, 0]
            )

            estimated_vy = float(
                kf.x[3, 0]
            )


            # ----------------------------------------------
            # Project prediction
            # ----------------------------------------------

            pred_px, pred_py = world_to_fpa(
                estimated_x,
                estimated_y,
                camera
            )


            error_x = (
                pred_px
                - camera.width / 2
            )

            error_y = (
                pred_py
                - camera.height / 2
            )


            # ----------------------------------------------
            # Angular error
            # ----------------------------------------------

            angular_error_x = (
                error_x
                * camera.hfov
                / camera.width
            )

            angular_error_y = (
                error_y
                * camera.vfov
                / camera.height
            )


            # ----------------------------------------------
            # Velocity feed-forward
            # ----------------------------------------------

            angular_vx, angular_vy = (
                angular_velocity_53h(
                    estimated_x,
                    estimated_y,
                    estimated_vx,
                    estimated_vy,
                    camera
                )
            )


            # ----------------------------------------------
            # Predictive controller
            # ----------------------------------------------

            pan_rate = (
                KP_ANGLE
                * angular_error_x
                +
                KV_ANGLE
                * angular_vx
            )

            tilt_rate = (
                KP_ANGLE
                * angular_error_y
                +
                KV_ANGLE
                * angular_vy
            )


            # ----------------------------------------------
            # Physical limits
            # ----------------------------------------------

            pan_rate = np.clip(
                pan_rate,
                -camera.max_pan_speed,
                camera.max_pan_speed
            )

            tilt_rate = np.clip(
                tilt_rate,
                -camera.max_tilt_speed,
                camera.max_tilt_speed
            )


        else:

            # ----------------------------------------------
            # Acquisition search
            # ----------------------------------------------

            pan_rate, tilt_rate = (
                search.update(
                    camera,
                    DT_CONTROL
                )
            )


        # ====================================================
        # CAMERA UPDATE
        # ====================================================

        camera.move(
            pan_rate,
            tilt_rate,
            DT_CONTROL
        )


        # ====================================================
        # TRUE TRACKING ERROR
        # ====================================================

        actual_px, actual_py = world_to_fpa(
            target_x,
            target_y,
            camera
        )

        error = np.sqrt(
            (actual_px - camera.width / 2)**2
            +
            (actual_py - camera.height / 2)**2
        )

        errors.append(error)


        if error <= LOCK_THRESHOLD:

            lock_count += 1


    # ========================================================
    # CALCULATE RESULTS
    # ========================================================

    elapsed = time.time() - start_time

    avg_error = np.mean(errors)

    max_error = np.max(errors)

    final_error = errors[-1]

    lock_percentage = (
        lock_count
        /
        total_frames
        * 100
    )

    detection_percentage = (
        detection_count
        /
        max(1, vision_frames)
        * 100
    )

    visible_detection_percentage = (
        detection_count
        /
        max(
            1,
            int(
                np.ceil(
                    visible_frames
                    /
                    YOLO_INTERVAL
                )
            )
        )
        * 100
    )

    avg_detection_error = (
        np.mean(detection_errors)
        if detection_errors
        else np.nan
    )

    runtime_fps = (
        total_frames
        /
        elapsed
    )


    return {
        "Trajectory": motion_type,
        "Final Error (px)": final_error,
        "Average Error (px)": avg_error,
        "Maximum Error (px)": max_error,
        "Lock <=10px (%)": lock_percentage,
        "YOLO Detection (%)": detection_percentage,
        "Visible Detection (%)": visible_detection_percentage,
        "Avg Detection Error (px)": avg_detection_error,
        "Visible Frames": visible_frames,
        "YOLO Frames": vision_frames,
        "Max Consecutive Miss": max_miss,
        "Runtime FPS": runtime_fps
    }


# ============================================================
# RUN ALL TRAJECTORIES
# ============================================================

print("=" * 70)
print("STEP 53H: CORRECTED MULTI-TRAJECTORY TRACKING BENCHMARK")
print("=" * 70)

results_53h = []

for trajectory in TRAJECTORIES_53H:

    print()
    print(
        f"Running: {trajectory.upper()} ..."
    )

    result = benchmark_trajectory_53h(
        trajectory
    )

    results_53h.append(
        result
    )

    print(
        f"  Final Error : "
        f"{result['Final Error (px)']:.2f} px"
    )

    print(
        f"  Average     : "
        f"{result['Average Error (px)']:.2f} px"
    )

    print(
        f"  Max Error   : "
        f"{result['Maximum Error (px)']:.2f} px"
    )

    print(
        f"  Lock <=10px : "
        f"{result['Lock <=10px (%)']:.2f}%"
    )

    print(
        f"  YOLO Detect : "
        f"{result['YOLO Detection (%)']:.2f}%"
    )

    print(
        f"  Visible Det : "
        f"{result['Visible Detection (%)']:.2f}%"
    )

    print(
        f"  Runtime FPS : "
        f"{result['Runtime FPS']:.2f}"
    )


# ============================================================
# SUMMARY
# ============================================================

results_53h_df = pd.DataFrame(
    results_53h
)

print()
print("=" * 70)
print("STEP 53H SUMMARY")
print("=" * 70)

display(
    results_53h_df.round(2)
)

print("=" * 70)


# ======================================================================
# ORIGINAL NOTEBOOK CELL 334
# ======================================================================
# ============================================================
# STEP 53I: ADAPTIVE KALMAN FILTER CONFIGURATION
# ============================================================

from filterpy.kalman import KalmanFilter


def create_adaptive_kalman(
    process_noise=0.1,
    measurement_noise=4.0,
    initial_position=(1005.0, 1000.0)
):

    kf = KalmanFilter(
        dim_x=4,
        dim_z=2
    )

    # --------------------------------------------------------
    # State:
    # [x, y, vx, vy]
    # --------------------------------------------------------

    kf.x = np.array([
        [initial_position[0]],
        [initial_position[1]],
        [0.0],
        [0.0]
    ], dtype=float)


    # --------------------------------------------------------
    # State transition
    # --------------------------------------------------------

    kf.F = np.array([
        [1, 0, DT_CONTROL, 0],
        [0, 1, 0, DT_CONTROL],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)


    # --------------------------------------------------------
    # Measurement model
    # --------------------------------------------------------

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)


    # --------------------------------------------------------
    # Measurement noise
    # --------------------------------------------------------

    kf.R = np.eye(2) * measurement_noise


    # --------------------------------------------------------
    # Process noise
    # --------------------------------------------------------

    kf.Q = np.eye(4) * process_noise


    # --------------------------------------------------------
    # Initial uncertainty
    # --------------------------------------------------------

    kf.P = np.eye(4) * 100.0


    return kf


print("=" * 70)
print("STEP 53I: ADAPTIVE KALMAN FILTER CONFIGURATION")
print("=" * 70)

print("State              : [x, y, vx, vy]")
print("Control frequency  : 30 Hz")
print("Measurement noise  : 4.0")
print("Process noise      : configurable")
print()
print("Purpose:")
print("  ✓ Faster velocity adaptation")
print("  ✓ Better curved-motion prediction")
print("  ✓ Better figure-8 tracking")
print("  ✓ Better response to random motion")
print("=" * 70)


# ======================================================================
# ORIGINAL NOTEBOOK CELL 354
# ======================================================================
# ============================================================
# STEP 57A: RESTORE BOUNDED TARGET
# ============================================================

print("=" * 70)
print("STEP 57A: RESTORING BOUNDED TARGET")
print("=" * 70)


class BoundedTarget:

    def __init__(
        self,
        motion_type="straight",
        center=(1005.0, 1000.0),
        amplitude_x=25.0,
        amplitude_y=18.0,
        speed=40.0,
        phase=0.0
    ):

        self.motion_type = motion_type
        self.cx = float(center[0])
        self.cy = float(center[1])

        self.amplitude_x = float(amplitude_x)
        self.amplitude_y = float(amplitude_y)

        self.speed = float(speed)
        self.phase = float(phase)

        self.t = 0.0

        self.x = self.cx + self.amplitude_x
        self.y = self.cy

        self.W = 2000.0
        self.H = 2000.0

        self.margin = 100.0

        # For random motion
        self.random_direction = self.phase
        self.random_timer = 0.0

    def update(self, dt):

        self.t += dt

        # ====================================================
        # CIRCULAR
        # ====================================================

        if self.motion_type == "circular":

            radius = max(
                1.0,
                (abs(self.amplitude_x) +
                 abs(self.amplitude_y)) / 2.0
            )

            omega = self.speed / radius

            angle = (
                self.phase +
                omega * self.t
            )

            self.x = (
                self.cx +
                self.amplitude_x *
                np.cos(angle)
            )

            self.y = (
                self.cy +
                self.amplitude_y *
                np.sin(angle)
            )

        # ====================================================
        # FIGURE-8
        # ====================================================

        elif self.motion_type == "figure8":

            radius = max(
                1.0,
                (abs(self.amplitude_x) +
                 abs(self.amplitude_y)) / 2.0
            )

            omega = self.speed / radius

            angle = (
                self.phase +
                omega * self.t
            )

            self.x = (
                self.cx +
                self.amplitude_x *
                np.sin(angle)
            )

            self.y = (
                self.cy +
                self.amplitude_y *
                np.sin(angle) *
                np.cos(angle)
            )

        # ====================================================
        # SPIRAL
        # ====================================================

        elif self.motion_type == "spiral":

            period = 12.0

            phase = (
                self.t % period
            ) / period

            scale = (
                0.2 +
                0.8 * phase
            )

            angle = (
                self.phase +
                self.speed *
                self.t /
                max(self.amplitude_x, 1.0)
            )

            self.x = (
                self.cx +
                self.amplitude_x *
                scale *
                np.cos(angle)
            )

            self.y = (
                self.cy +
                self.amplitude_y *
                scale *
                np.sin(angle)
            )

        # ====================================================
        # RANDOM
        # ====================================================

        elif self.motion_type == "random":

            self.random_timer += dt

            if self.random_timer >= 1.0:

                self.random_timer = 0.0

                self.random_direction = np.random.uniform(
                    0,
                    2 * np.pi
                )

            self.x += (
                self.speed *
                np.cos(self.random_direction) *
                dt
            )

            self.y += (
                self.speed *
                np.sin(self.random_direction) *
                dt
            )

            # Boundaries
            if self.x < self.margin:
                self.x = self.margin
                self.random_direction = np.pi - self.random_direction

            if self.x > self.W - self.margin:
                self.x = self.W - self.margin
                self.random_direction = np.pi - self.random_direction

            if self.y < self.margin:
                self.y = self.margin
                self.random_direction = -self.random_direction

            if self.y > self.H - self.margin:
                self.y = self.H - self.margin
                self.random_direction = -self.random_direction

        # ====================================================
        # STRAIGHT
        # ====================================================

        else:

            angle = self.phase

            self.x += (
                self.speed *
                np.cos(angle) *
                dt
            )

            self.y += (
                self.speed *
                np.sin(angle) *
                dt
            )

            if self.x < self.margin or self.x > self.W - self.margin:
                self.x = np.clip(
                    self.x,
                    self.margin,
                    self.W - self.margin
                )

            if self.y < self.margin or self.y > self.H - self.margin:
                self.y = np.clip(
                    self.y,
                    self.margin,
                    self.H - self.margin
                )

        return self.x, self.y


# ------------------------------------------------------------
# Restore FAIR_CENTER if necessary
# ------------------------------------------------------------

try:
    FAIR_CENTER
except NameError:
    FAIR_CENTER = (1005.0, 1000.0)


print("✓ BoundedTarget restored")
print(f"✓ FAIR_CENTER = {FAIR_CENTER}")
print("✓ Circular trajectory available")
print("✓ Figure-8 trajectory available")
print("✓ Random trajectory available")


# ======================================================================
# ORIGINAL NOTEBOOK CELL 358
# ======================================================================
# ============================================================
# STEP 57D: RESTORE CONSISTENT FPA RENDERER
# ============================================================

print("=" * 70)
print("STEP 57D: RESTORING CONSISTENT FPA RENDERER")
print("=" * 70)


def render_consistent_fpa(
    target_x,
    target_y,
    camera,
    beacon_size=10
):
    """
    Render the beacon using the same validated
    world_to_fpa() geometry.
    """

    frame = np.zeros(
        (camera.height, camera.width),
        dtype=np.uint8
    )

    # World → FPA
    px, py = world_to_fpa(
        target_x,
        target_y,
        camera
    )

    # Check visibility
    visible = (
        0 <= px < camera.width
        and
        0 <= py < camera.height
    )

    if visible:

        x = int(round(px))
        y = int(round(py))

        radius = max(
            1,
            beacon_size // 2
        )

        cv2.circle(
            frame,
            (x, y),
            radius,
            255,
            -1
        )

        # Slight optical blur
        frame = cv2.GaussianBlur(
            frame,
            (0, 0),
            1.5
        )

    return frame, px, py


print("✓ render_consistent_fpa() restored")
print("✓ Uses validated world_to_fpa()")
print("✓ Beacon size = 10 px")
print("✓ Gaussian optical blur enabled")
