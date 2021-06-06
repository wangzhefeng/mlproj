

set.seed(4)

# 读取数据、添加异常值
data = read.csv("F:/personal/timeseries/data/webTraffic.csv")
days = as.numeric(data$Visite)
for (i in 1:45) {
    pos = floor(runif(1, 1, 50))
    days[i * 15 + pos] = days[i * 15 + pos] ^ 1.2
}
days[510 + pos] = 0
plot(as.ts(days))

# 季节性检测(傅里叶变换)




# ===========================
# 移动平均分解异常检测
# ===========================
# 时间序列(乘法)分解
library(FBN)
decomposed_days = decompose(ts(days, frequency = 7), type = "multiplicative")
plot(decomposed_days)


# 对于随即噪声，应用正太分布检测异常
random = decomposed_days$random
min = mean(random, na.rm = TRUE) - 4 * sd(random, na.rm = TRUE)
max = mean(random, na.rm = TRUE) + 4 * sd(random, na.rm = TRUE)
plot(as.ts(random), ylim = c(-0.5, 2.5))
abline(h = min, col = "#e15f3f", lwd = 2)
abline(h = max, col = "#e15f3f", lwd = 2)


# ===========================
# 移动中位数分解
# ===========================

