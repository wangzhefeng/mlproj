library(DMwR)
library(tidyverse)
library(lattice)


# Miss value in data
algae[!complete.cases(algae), ]
nrow(algae[!complete.cases(algae), ])

sapply(algae, function(x) sum(is.na(x)))
apply(algae, 2, function(x) sum(is.na(x)))

apply(algae, 1, function(x) sum(is.na(x)))


# delete the rows which have miss values in algae
algae = na.omit(algae)


# delete the rows which have many miss values
data(algae)
algae = algae[-c(62, 199), ]


data(algae)
manyNAs(algae, 0.2)
algae = algae[-manyNAs(algae), ]



# impute the miss values with center values
data(algae)
algae[48, "mxPH"] = mean(algae$mxPH, na.rm = TRUE)

algae[is.na(algae$Chla), "Chla"] = median(algae$Chla, na.rm = TRUE)

data(algae)
algae = algae[-manyNAs(algae), ]
algae
algae = centralImputation(algae)



# 通过变量之间的相关关系来填充缺失值
data(algae)
cor(algae[, 4:18], use = "complete.obs")
symnum(cor(algae[, 4:18], use = "complete.obs"))

algae = algae[-manyNAs(algae), ]
model = lm(PO4 ~ oPO4, data = algae)
algae[28, "PO4"] = 42.897 + 1.293 * algae[28, "oPO4"]

data(algae)
algae = algae[-manyNAs(algae), ]
fillPO4 = function(oP) {
    if(is.na(oP)){
        return(NA)
    } else {
        return(42.897 + 1.293 * oP)
    }
}
algae[is.na(algae$PO4), "PO4"] = sapply(algae[is.na(algae$PO4), "oPO4"], fillPO4)


# 
library(lattice)
algae$season = factor(algae$season, 
                      levels = c("spring", "summer", "autumn", "winter"))
histogram(~mxPH | season, data = algae)
histogram(~mxPH | size, data = algae)
histogram(~mxPH | size * speed, data = algae)

stripplot(size ~ mxPH | speed, data = algae, jitter = TRUE)

# 通过探索数据行之间的相似性填补缺失值
data(algae)
algae = algae[-manyNAs(algae), ]

algae = knnImputation(algae, k = 10)
algae = knnImputation(algae, k = 10, meth = "median")


