library(fpp)
library(Ecdat)

data("ausbeer")
data("AirPassengers")
timeseries_beer = tail(head(ausbeer, 17 * 4 + 2), 17 * 4 - 4)
timeseries_air = AirPassengers

plot(as.ts(timeseries_beer))
plot(as.ts(timeseries_air))

