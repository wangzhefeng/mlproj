library(DMwR)
library(rpart)
library(tidyverse)


data(algae)
algae = algae[-manyNAs(algae), ]
rt_a1 = rpart(a1 ~ ., data = algae[, 1:12])

# results
rt_a1
summary(rt_a1)

# method 1
plot(rt_a1)
text(rt_a1)

# method 2
prettyTree(rt_a1)


# jianzhi
printcp(rt_a1)

rt2_a1 = prune(rt_a1, cp = 0.08)
rt2_a1

DMwR::rpartXse(a1 ~ ., data = algae[, 1:12])


first_tree = rpart(a1 ~ ., data = algae[, 1:12])
snip.rpart(first_tree, c(4, 7))

