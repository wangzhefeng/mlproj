library(DMwR)
library(lattice)

minO2 = equal.count(na.omit(algae$mnO2), number = 4, overlap = 1 / 5)
stripplot(season ~ a3|minO2, data = algae[!is.na(algae$mnO2), ])
