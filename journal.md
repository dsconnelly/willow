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

#### 31 August 2022
- I met with Ed and Zihan. We discussed the offline performance of Zihan's SVR
models trained on MiMA data and possible analyses to conduct before online
testing. I showed Ed the mean state plots and discussed ways to better
understand the failuer of models taking `Nsq` to produce a QBO.
- I added code to compute and plot Gini importances for tree models, in case
the `Nsq` models were splitting many nodes with `Nsq` in a pathological way that
would not show up in the Shapley values.
- That does need seem to be the case; however, I realized that using any `Nsq`
information from levels potentially below the source level is problematic for
forests. So I modified the `make_datasets` function to replace those `Nsq`
levels with zeros and trained a new model. I think this will resolve the issue.