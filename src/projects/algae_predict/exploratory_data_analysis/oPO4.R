library(lagae)

boxplot(algae$oPO4, ylab = "Orthophoshate (oPO4)")
rug(jitter(algae$oPO4), side = 2)
abline(h = mean(algae$oPO4, na.rm = TRUE), 
       lty = 2, 
       col = "red")
