# add_z_to_file.R
# Author: Daniel Zurawski
# Modify the in_filename file by adding a linear-sloping z component and then
# save this new file to out_filename.
# Take a dataframe with columns (event_id, cluster_id, x, y)
# Save a dataframe with columns (event_id, cluster_id, r, phi, z).

require("tidyr")
require("dplyr")

in_filename   <- "../data/sets/generated.csv"
out_filename  <- "../data/sets/UNIF50-50000E-Z.csv"
z.bounds      <- c(-200, 200) # What should the min and max z values be?
initial.frame <- read.csv(in_filename) %>%
                    mutate(phi = atan2(y, x)) %>%                
                    mutate(r = round(sqrt(x * x + y * y), 6))
r.max         <- max(initial.frame$r)
eta.bounds    <- c(atan(r.max / z.bounds[1] - 12 * pi),
                   atan(r.max / z.bounds[1]),
                   atan(r.max / z.bounds[2]),
                   atan(r.max / z.bounds[2] + 12 * pi))

stopifnot(eta.bounds[1] < eta.bounds[2]) # runif doesn't work correctly
stopifnot(eta.bounds[3] < eta.bounds[4]) # if these conditions are false.

initial.frame <-initial.frame %>%
    group_by(event_id, cluster_id) %>%
    mutate(eta = sample(c(runif(1, eta.bounds[1], eta.bounds[2]),
                          runif(1, eta.bounds[3], eta.bounds[4])), 1)) %>%
    ungroup() %>%
    mutate(z = ((r) / tan(eta))) %>%
    arrange(event_id, cluster_id, r) %>%
    select(event_id, cluster_id, r, phi, z)

write.csv(
    initial.frame,
    out_filename,
    row.names = TRUE
)

print(sort(unique(initial.frame$r)))
print(min(abs(initial.frame$z)))
print(max(abs(initial.frame$z)))
