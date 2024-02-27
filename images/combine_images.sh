#!/bin/sh

# This script combines some of the images from the documentation into
# a single large image suitable for conference submissions that only
# allow a single image.

mkdir -p build

# Synthetic data, with intermediate subtitles.
montage -tile 2x1 -title "When scatter plots are ineffective or murky..." -geometry +32+32 -border 8 y_vs_x3.png y_vs_x2.png build/scatter.png
montage -tile 2x1 -title "Impact charts clearly identify the impact of a single feature." -geometry +32+32 -border 8 x3_impact.png x2_impact.png build/synth_impact.png

#montage -tile 1x2 -geometry +32+32 scatter.png synth_impact.png synth.png
convert build/scatter.png build/synth_impact.png -gravity south -append build/synth.png

# Real data
montage -tile 2x1 -title "A real world example from an analysis of the impact of income, race, and ethnicity on eviction rates." -geometry +32+32 -border 8 13089-income.png 13089-black.png build/real-impact.png


# QR code

convert qrcode.png -bordercolor white -border 514x10 -gravity South -pointsize 24 -annotate 0 "Scan or visit https://github.com/vengroff/impactchart to learn more." build/qrannotated.png

# Combine them all

montage -tile 1x3 -pointsize 48 -title "An Introduction to Impact Charts" -geometry +32+32 build/synth.png build/real-impact.png build/qrannotated.png build/poster.png








