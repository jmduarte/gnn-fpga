################################################################################
add_z_to_file.R

"Add a linear component to the 2D tracks. This linear component is randomly
selected. Here is some pseudo-code to illustrate this process.

For each track in a set of 2D tracks:
	1. Choose a random slope constant, m.
	2. Define a mapping from radius to z coordinate, z[r] = m * r.
	3. Apply this mapping to each hit in the track.
	4. This applied mapping contains the z values for each hit.
"

################################################################################
generate.py

"Generate events using randomly sampled tracks from the RAMP data set.
This file is used to obtain a list of events to train a model."

################################################################################
GENERATE-ACTS.ipynb

"This notebook applies the generation functions in generate.py to generate
events."

################################################################################
