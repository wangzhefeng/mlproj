library(DMwR)
library(tidyverse)

# 识别变量中的离群值
plot(algae$NH4, xlab = "")
abline(h = mean(algae$NH4, na.rm = TRUE), lty = 1)
abline(h = mean(algae$NH4, na.rm = TRUE) + sd(algae$NH4, na.rm = TRUE), lty = 2)
abline(h = median(algae$NH4, na.rm = TRUE), lty = 3)

# Method 1
# identify(algae$NH4)
clicked.lines = identify(algae$NH4)
algae[clicked.lines, ]


# Method 2
algae[algae$NH4 > 19000, ]
algae[!is.na(algae$NH4) & algae$NH4 > 19000, ]

# or 
algae %>% filter(NH4 > 19000)
algae %>% filter(is.na(NH4) & NH4 > 19000)
