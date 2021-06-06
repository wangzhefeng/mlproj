library(DMwR)

# Histogram plot, Density plot and Normal Q-Q plot of mxPH
par(mfrow = c(1, 2))

hist(algae$mxPH, 
     prob = TRUE, 
     xlab = "", 
     main = "Histogram of maximum pH value",
     ylim = 0:1)
lines(density(algae$mxPH, na.rm = TRUE))
rug(jitter(algae$mxPH))

# Normal Q-Q 给出了变量分位数和正态分布的理论分位数的散点图
# 并给出了正态分布的95%置信区间的袋带状图
library(car)
qqPlot(algae$mxPH, main = "Normal QQ plot of maximum pH") 

par(mfrow = c(1, 1))
