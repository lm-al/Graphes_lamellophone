# Spectrogram for music mp3 or mp4 script

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ============================================================

# AUDIO FILE

# ============================================================

audio_file = (
    "montage2/music_letter_crop.mp4"
)

output_file = (
    "spectrogram_letter_c.eps"
)

# ============================================================

# SPECTROGRAM PARAMETERS

# ============================================================

n_fft = 4096
hop_length = 512

# Duration of the analyzed window

T = 15.33

# IMPORTANT:

# The beginning of our experiment corresponds to 10 s
# after the actual beginning of the music.

T_START = 0

# ============================================================

# FIGURE PARAMETERS

# ============================================================

# EXACTLY THE SAME AS IN RESLICE

# ratio 4:1
FIG_WIDTH = 10.5
FIG_HEIGHT = 2.625

AX_WIDTH = 8.0
AX_HEIGHT = 2.0

LEFT = 1.10
BOTTOM = 0.45

CBAR_X = 9.30
CBAR_WIDTH = 0.30

"""
# ratio 3:1
FIG_WIDTH = 8.5
FIG_HEIGHT = 2.625

AX_WIDTH = 6.0
AX_HEIGHT = 2.0

LEFT = 1.10
BOTTOM = 0.45

CBAR_X = 7.30
CBAR_WIDTH = 0.30
"""

# ============================================================

# LOAD AUDIO FILE

# ============================================================

y, sr = librosa.load(
    audio_file,
    sr=None,
    mono=True
)

# ============================================================

# SHORT-TIME FOURIER TRANSFORM

# ============================================================

D = librosa.stft(
    y,
    n_fft=n_fft,
    hop_length=hop_length,
    window="hann",
    center=False
)

# Convert to decibels

S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

# ============================================================

# TIME VALUES ASSOCIATED WITH THE SPECTROGRAM COLUMNS

# ============================================================

times = librosa.times_like(
    S_db,
    sr=sr,
    hop_length=hop_length
)

# ============================================================

# SELECT THE TIME WINDOW

# ============================================================

# We keep only:
#
# music: 10.0 s --> 26.4 s
#
# which becomes:
#
# displayed axis: 0 s --> 16.4 s

mask = (
    (times >= T_START) &
    (times <= T_START + T)
)

S_db_window = S_db[:, mask]

times_window = times[mask] - T_START

# ============================================================

# FIGURE

# ============================================================

fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

# ------------------------------------------------------------

# Main axis

# ------------------------------------------------------------

ax = fig.add_axes([
    LEFT / FIG_WIDTH,
    BOTTOM / FIG_HEIGHT,
    AX_WIDTH / FIG_WIDTH,
    AX_HEIGHT / FIG_HEIGHT
])

# ============================================================

# SPECTROGRAM

# ============================================================

img = librosa.display.specshow(
    S_db_window,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="linear",
    cmap="gray",
    ax=ax,
    x_coords=times_window
)

# ============================================================

# X-AXIS: EXPERIMENTAL TIME

# ============================================================

# IMPORTANT:
#
# The axis remains from 0 to 16.4 s.
#
# 0 s = 10 s in the audio file
# 2 s = 12 s in the audio file
# 4 s = 14 s in the audio file
# ...
# 16 s = 26 s in the audio file

ax.set_xlim(
    0,
    T
)

# Ticks every 2 s

x_ticks = np.arange(
    0,
    T + 0.1,
    2
)

ax.set_xticks(
    x_ticks
)

ax.set_xticklabels(
    [f"{t:.0f}" for t in x_ticks]
)

ax.set_xlabel(
    "Time (seconds)"
)

# ============================================================

# Y-AXIS: FREQUENCY

# ============================================================

ax.set_ylim(
    540,
    2200
)

y_ticks = np.arange(
    540,
    2200,
    200
)

ax.set_yticks(
    y_ticks
)

ax.set_ylabel(
    "Frequency (Hz)"
)

# ============================================================

# COLORBAR

# ============================================================

cax = fig.add_axes([
    CBAR_X / FIG_WIDTH,
    BOTTOM / FIG_HEIGHT,
    CBAR_WIDTH / FIG_WIDTH,
    AX_HEIGHT / FIG_HEIGHT
])

cbar = fig.colorbar(
    img,
    cax=cax
)

cbar.set_label(
    "Amplitude (dB)"
)

# ============================================================

# SAVE

# ============================================================

fig.savefig(
    output_file,
    format="eps",
    dpi=600

    # DO NOT use bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Spectrogram saved: {output_file}"
)

print(
    f"Time window used in the music: "
    f"{T_START:.1f} s --> {T_START + T:.1f} s"
)
