I ought to keep better track of what I do on a day-to-day basis. I'll try to do so here.

#### 29 August 2022
- I added the `plot_climatology` function to `willow`, which allows me to take cursory looks at the mean states of coupled MiMA runs.
- I discussed `cos(lat)` sampling with Zihan.
- I realized that I failed to output buoyancy frequency in the two online runs I started last Friday (`mubofo-mean-T` and `mubofo-mean-Nsq`) &mdash; problematic, because the point of those runs was to look at mean states and see if there's an obvious bias that might cause GWPs taking `Nsq` to produce worse QBOs. So I resubmitted those two runs with the appropriate `diag_table`.

#### 30 August 2022
- I added significant functionality to `_climatology.py`, first adding separate
functions to plot tropical, vertical, and zonal means, but then deciding to
combine those functions into one `plot_climatologies` function.
- I submitted jobs to compare u, v, T, and N, viewed through all three kinds of
mean, in the outputs from `control`, `mubofo-mean-T`, and `mubofo-mean-Nsq`.