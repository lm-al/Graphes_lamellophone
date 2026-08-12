# RESLICE script for the data resliced by imageJ

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================

# DIRECTORIES

# ============================================================

input_dir = Path("reslice/avi")
output_dir = Path("reslice/eps_3:1")

output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================

# DATA PARAMETERS

# ============================================================

N = 765          # Number of frames after cropping: 765 for Elise, 770 for Turk
T = 16.4*N/818   # Video duration (s): original registered data = 16.4 s, hence 818 in ImageJ
print(T, T+16.4*33/818)

# ============================================================

# FIGURE PARAMETERS

# ============================================================

# Same figure size as in spectra.py

FIG_WIDTH = 8.5
FIG_HEIGHT = 2.625

AX_WIDTH = 6.0
AX_HEIGHT = 2.0

LEFT = 1.10
BOTTOM = 0.45

CBAR_X = 7.30
CBAR_WIDTH = 0.30

# ============================================================

# AVI FILE PROCESSING

# ============================================================

avi_files = sorted(input_dir.glob("*.avi"))

for avi_path in avi_files:

    print(f"Processing: {avi_path.name}")

    # --------------------------------------------------------
    # Open the video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(avi_path))

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        frames.append(
            frame.astype(np.float64)
        )

    cap.release()

    # --------------------------------------------------------
    # Check the number of frames
    # --------------------------------------------------------

    if len(frames) == 0:

        print(
            "  -> Empty video, file skipped."
        )

        continue

    if len(frames) != N:

        print(
            f"  -> WARNING: "
            f"{len(frames)} frames found "
            f"(expected: {N})"
        )

    else:

        print(
            f"  -> {N} frames detected."
        )

    # --------------------------------------------------------
    # Stack the images
    # --------------------------------------------------------

    stack = np.stack(frames)

    # --------------------------------------------------------
    # Temporal average
    # --------------------------------------------------------

    mean_image = np.mean(
        stack,
        axis=0
    )

    # Image dimensions
    height, width = mean_image.shape

    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    # --------------------------------------------------------
    # Main axis
    # --------------------------------------------------------

    ax = fig.add_axes([
        LEFT / FIG_WIDTH,
        BOTTOM / FIG_HEIGHT,
        AX_WIDTH / FIG_WIDTH,
        AX_HEIGHT / FIG_HEIGHT
    ])

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    im = ax.imshow(
        mean_image,
        cmap="gray",
        origin="lower",
        aspect="auto",

        # Horizontal correspondence:
        # 0 -> 16.4 seconds
        extent=[
            0,
            T,
            0,
            height
        ]
    )

    # ========================================================
    # Y-AXIS: CANTILEVERS
    # ========================================================

    n_lamelles = 18

    y_ticks = (
        np.linspace(
            0,
            height,
            n_lamelles,
            endpoint=False
        )
        + height / (2 * n_lamelles)
    )

    # Only even numbers
    y_ticks = y_ticks[1::2]

    y_labels = np.arange(
        2,
        19,
        2
    )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_ylabel(
        "Cantilever number"
    )

    # ========================================================
    # X-AXIS: TIME
    # ========================================================

    ax.set_xlim(
        0,
        T
    )

    ticks_sec = np.arange(
        0,
        T + 0.1,
        2
    )

    ax.set_xticks(
        ticks_sec
    )

    ax.set_xticklabels(
        [f"{t:.0f}" for t in ticks_sec]
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    # ========================================================
    # COLORBAR
    # ========================================================

    cax = fig.add_axes([
        CBAR_X / FIG_WIDTH,
        BOTTOM / FIG_HEIGHT,
        CBAR_WIDTH / FIG_WIDTH,
        AX_HEIGHT / FIG_HEIGHT
    ])

    cbar = fig.colorbar(
        im,
        cax=cax
    )

    cbar.set_label(
        "Mean Intensity"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        output_dir /
        f"{avi_path.stem}_3_1.eps"
    )

    fig.savefig(
        output_path,
        format="eps",
        dpi=300

        # IMPORTANT:
        # DO NOT use bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"  -> Saved: {output_path}"
    )

print("\nRESLICE processing completed.")
