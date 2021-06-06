library(DMwR)
library(dplyr)

# data 
data(algae)
algae <- algae[-manyNAs(algae), ]
clean_algae = knnImputation(algae, k = 10)

# built the regression model 
lm_a1 = lm(a1 ~ ., data = clean_algae[, 1:12])
summary(lm_a1)

anova(lm_a1)

lm2_a1 = update(lm_a1, . ~ . - season)
summary(lm2_a1)

anova(lm2_a1)

lm3_a1 = update(lm2_a1, .~. - Chla)
summary(lm3_a1)

anova(lm2_a1, lm3_a1)


# step
final_lm_back = step(lm_a1, direction = "backward")
summary(final_lm_back)
final_lm_forward = step(lm_a1, direction = "forward")
final_lm_both = step(lm_a1, direction = c("both", "backward", "forward"))





