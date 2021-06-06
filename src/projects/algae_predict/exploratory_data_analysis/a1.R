library(DMwR)


library(lattice)
bwplot(size ~ a1, data = algae, ylab = "River Size", xlab = "Algal A1")

library(Hmisc)
bwplot(size ~ a1,
       data = algae, 
       panel = panel.bpplot, 
       probs = seq(0.01, 0.49, by = 0.01),
       datadendity = TRUE,
       ylab = "River Size",
       xlab = "Algal A1")

