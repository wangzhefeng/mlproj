# *********************************************************************
# current question:
# *********************************************************************
# 为什么不先进行缺失值填充再进行特征工程？
# 正则表达式和正则表达式函数？
# 缺失值填充问题？
# *********************************************************************
# Analysis flow:
# *********************************************************************
# Load R libraries 
# Load training and testing datasets
# 异常值处理;
# 缺失值处理;
# 正态性检验及处理(因变量);
# Feature Engineering
# 数据分割(CV)
# 建模
# 模型评估与比较;
# *********************************************************************
# models
# *********************************************************************
# CART
# logsitic
# bagging
# random forest
# GBDT



# *********************************************************************
# 载入R包
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(scales)) install.packages("scales")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(readr)) install.packages("readr")
if(!require(mice)) install.packages("mice")
if(!require(caret)) install.packages("caret")
if(!require(rpart)) install.packages("rpart")
if(!require(gbm)) install.packages("gbm")
if(!require(randomForest)) install.packages("randomForest")
if(!require(odbc)) install.packages("odbc")
if(!require(DBI)) install.packages("DBI")
if(!require(Amelia)) install.packages("Amelia")

# *********************************************************************
# 载入训练数据和测试数据
conn = DBI::dbConnect(odbc::odbc(),
                      Driver = "SQL Server",
                      Server = "WANGZF-PC",
                      Database = "tinker",
                      UID = "tinker.wang",
                      PWD = "alvin123",
                      Port = 1433)

train = dbReadTable(conn, "titanic_train")
test = dbReadTable(conn, "titanic_test")
# train = readr::read_csv(file = "E:/machinelearning/pro_Titanic/data/train.csv")
# test = readr::read_csv(file = "E:/machinelearning/pro_Titanic/data/test.csv")
all = dplyr::bind_rows(train, test)


# *********************************************************************
# 数据探索性分析
# *********************************************************************
# 数据
str(all)

# 数据概览
summary(all)
Hmisc::describe(all)


# 数据类型检查
numCols = c("Age", "SibSp", "Parch", "Fare")
cateCols = c("Survived", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked")
orderedCateCols = c("Pclass")

# 数据缺失值检查
md.pattern(all)
sapply(all, function(x) sum(is.na(x)))


# 异常值检查，处理
# Age < 1
all %>% filter(Age < 1 & Fare == '151.5500')

all %>% 
    filter(Age < 1) %>% 
    ggplot(mapping = aes(x = Age, y = Fare)) +
    geom_point(mapping = aes(colour = factor(Pclass)), size = 2)


# Fare == 0
all %>% filter(Fare == 0)

# all %>% 
#     filter(Fare > 0) %>%
#     group_by(Pclass) %>%
#     summarise(Fare_Med = median(Fare), 
#               Fare_Mean = mean(Fare))
# 
# all[all$Fare == 0 & all$Pclass == 1 & !is.na(all$Fare), ]$Fare = 
#     median(all[all$Fare > 0 & all$Pclass == 1, ]$Fare, na.rm = TRUE)
# all[all$Fare == 0 & all$Pclass == 2 & !is.na(all$Fare), ]$Fare = 
#     median(all[all$Fare > 0 & all$Pclass == 2, ]$Fare, na.rm = TRUE)
# all[all$Fare == 0 & all$Pclass == 3 & !is.na(all$Fare), ]$Fare = 
#     median(all[all$Fare > 0 & all$Pclass == 3, ]$Fare, na.rm = TRUE)


# *********************************************************************
# 特征工程 1
# *************************************************
# Problem 1: 能否根据乘客名字推断乘客之间的关系？
# *************************************************
# 1. Grab title from passenger names
all = all %>% dplyr::mutate(Title = gsub("(.*, )|(\\..*)", "", Name)) # (.*, )|(..*)
# Show title counts by sex (有些是军人)
table(all$Sex, all$Title)
# Title with very low cell counts to be combined to "rare" level
rare_title = c("Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir")
all$Title[all$Title == "Mlle"] = "Miss"
all$Title[all$Title == "Ms"] = "Miss"
all$Title[all$Title %in% c("Mme", "Lady", "Dona", "the Countess")] = "Mrs"
all$Title[all$Title == "Dr" & all$Sex == 'female'] = "Mrs"
all$Title[all$Title == "Dr" & all$Sex == "male"] = "Mr"
all$Title[all$Title %in% rare_title] = "Mr"
table(all$Sex, all$Title)

# 2. Grab surname from passenger name
all = all %>% 
    dplyr::mutate(Surname = sapply(Name, 
                                   function(x) strsplit(x, split = '[,.]')[[1]][1]))

# Unique Surnames
nlevels(factor(all$Surname))

family = all %>% 
    # select(Surname, Name, Age, Sex, Embarked, Pclass, Cabin, Ticket, SibSp, Parch) %>%
    arrange(Surname) %>%
    filter(SibSp != 0 | Parch != 0)
family

# *************************************************
# Problem 2: 一个家庭中的成员是否是一起沉入海水中还是游在一起？
# *************************************************
# Create a family size variable including the passenger themselves
all = all %>% dplyr::mutate(FamilySize = SibSp + Parch + 1)

# Create a family variable 
all = all %>% dplyr::mutate(Family = paste(Surname, FamilySize, sep = "_"))

# Visualize the relationship between family size & survival
ggplot(data = all[1:dim(train)[1], ], 
       mapping = aes(x = FamilySize, fill = factor(Survived))) +
    geom_bar(stat = "count", position = "dodge") +
    scale_x_continuous(breaks = c(1:11)) +
    labs(x = "Family Size")

# Discretize family size
all$FamilySizeD[all$FamilySize == 1] = "singleton"
all$FamilySizeD[all$FamilySize <= 4 & all$FamilySize > 1] = "small"
all$FamilySizeD[all$FamilySize > 4] = "large"

# Show family size by survival using a mosaic plot
mosaicplot(table(all$FamilySizeD, all$Survived), 
           main = "Family Size by Survival", 
           shade = TRUE)


# *************************************************
# Passenger cabin
# *************************************************
all = all %>% 
    dplyr::mutate(Deck = factor(sapply(Cabin, 
                                       function(x) strsplit(x, NULL)[[1]][1])))


# *********************************************************************
# 缺失值处理
# *********************************************************************
# *************************************************
# 缺失值检查
# *************************************************
md.pattern(all)

all %>% 
    sapply(function(x) sum(is.na(x))) %>% 
    knitr::kable()


missmap(all, main = "Titanic Data - Missings Map", col = c("yellow", "black"), legend = FALSE)
# *************************************************
# 分析存在缺失的预测变量，并填充
# *************************************************
# *************************************************
# Fare -- 1
# *************************************************
all %>% 
    filter(is.na(Fare)) %>%
    select(c(PassengerId, Pclass, Fare, Embarked))

Pclass3_EmbarkedS = all %>% 
    filter(Pclass == '3' & Embarked == 'S') %>%
    select(c(PassengerId, Pclass, Fare, Embarked))


ggplot(data = Pclass3_EmbarkedS, mapping = aes(x = Fare)) +
    geom_density(fill = "#99b6ff", alpha = 0.4) +
    geom_vline(mapping = aes(xintercept = median(Fare, na.rm = TRUE)),
               colour = "red",
               linetype = "dashed",
               lwd = 1) +
    scale_x_continuous(labels = dollar_format()) +
    theme_few()

all$Fare[1044] = median(all[all$Pclass == '3' & all$Embarked == "S", ]$Fare, 
                        na.rm = TRUE)
# *************************************************
# Embarked -- 2
# *************************************************
all %>% 
    select(c(Pclass, Fare, Embarked)) %>% 
    filter(is.na(Embarked))

None_Embarked_Nas = all %>% 
    select(c(PassengerId, Pclass, Fare, Embarked, Age)) %>%
    filter(!is.na(Embarked)) %>%
    arrange(Fare)


ggplot(data = None_Embarked_Nas,
       mapping = aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
    geom_boxplot() +
    geom_hline(mapping = aes(yintercept = 80), 
               colour = "red",
               linetype = "dashed",
               lwd = 1) +
    scale_y_continuous(labels = dollar_format()) +
    theme_calc()

all$Embarked[c(62, 830)] = "C"

# *************************************************
# Age -- 263
# method 1 -- rpart
# method 2 -- mice
# *************************************************
# mice
factor_var = c("PassengerId", "Pclass", "Sex", "Embarked", 
               "Title", "Surname", "Family", "FamilySizeD")
all[factor_var] = all %>%
    select(factor_var) %>%
    lapply(function(x) as.factor(x))
set.seed(129)
mice_mod = mice(all[, !names(all) %in% c("PassengerId", "Name", "Ticket", 
                                         "Cabin", "Family", "Surname", 
                                         "Survived")],
                method = "rf")
mice_output = complete(mice_mod)
mice_output

# rpart
library(rpart)
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                       data = all_data[!is.na(all_data$Age), ], 
                       method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age,
                                             all_data[is.na(all_data$Age), ])


op = par(mfrow = c(1, 2))
hist(all$Age, freq = FALSE, main = "Age: Original Data", 
     col = "darkgreen", ylim = c(0, 0.04))
hist(mice_output$Age, freq = FALSE, main = "Age: MICE Output",
     col = "lightgreen", ylim = c(0, 0.04))

all$Age = mice_output$Age
par(op)
# *************************************************
# Cabin -- 1014 Deck -- 1014
# *************************************************





# *********************************************************************
# 特征工程 2
# *********************************************************************
# Child : Age < 18
# Mother: Sex == "female" & Age >= 18 & Child > 0 & Title != "Miss"
ggplot(data = all[1:891, ], mapping = aes(x = Age, fill = factor(Survived))) +
    geom_histogram() +
    facet_grid(. ~ Sex) +
    theme_calc()


all = all %>% mutate(Child = ifelse(Age < 18, "Child", "Adult"))
all$Child = all$Child %>% factor()
table(all$Child, all$Survived)

all = all %>%
    mutate(Mother = ifelse(Sex == "female" & Parch > 0 & Age > 18 & Title != "Miss", 
                           "Mother", "Not Mother"))
all$Mother = all$Mother %>% factor()
table(all$Mother, all$Survived)

md.pattern(all)


# *********************************************************************
# 数据重编码
# *********************************************************************
# Create dummary variables
# method 1
dummies = dummyVars(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                    data = train)
train2 =  predict(dummies, newdata = train)
train2 = as.data.frame(train2)
train2 = cbind(train[, 1:2], train2)
head(train2)

# method 2
dummies = model.matrix(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                       data = train)
train3 = as.data.frame(dummies)
head(train3)





# *********************************************************************
# 数据分割 —— cross validation(10 folds)
# *********************************************************************
# Split training and testing data set
set.seed(123)
inTrain = sample(1:2, dim(train)[1], replace = TRUE, prob = c(0.8, 0.2))
training = train[inTrain == 1, ]
testing = train[inTrain == 2, ]
head(training)
dim(training)
dim(testing)

set.seed(123)
inTrain = createDataPartition(y = train$Survived, p = 0.8, list = FALSE)
training = train[inTrain, ]
testing = train[-inTrain, ]
head(training)
dim(training)
dim(testing)

set.seed(123)
inTrain = sample(1:2, dim(train2)[1], replace = TRUE, prob = c(0.8, 0.2))
training2 = train2[inTrain == 1, ]
testing2 = train2[inTrain == 2, ]
head(training2)
dim(training2)
dim(testing2)

set.seed(123)
inTrain = createDataPartition(y = train2$Survived, p = 0.8, list = FALSE)
training2 = train2[inTrain, ]
testing2 = train2[-inTrain, ]
head(training2)
dim(training2)
dim(testing2)

## 5-folds CV
set.seed(123)
folds = createFolds(y = train$Survived, k = 5, list = FALSE, returnTrain = TRUE)
table(folds)
sapply(folds, length)

## repeat 5-folds cv
set.seed(123)
folds = createMultiFolds(y = train$Survived, k = 5, list = FALSE, times = 3)
sapply(folds, length)

## Bootstrap
set.seed(123)
resamples = createResample(y = train$Survived, times = 10, list = TRUE)


train = all[1:891, ]
test = all[892:1309, ]

# *********************************************************************
# Modeling 
# *********************************************************************
# *************************************************
# Logistic Regression (LR)
# *************************************************




# *************************************************
# Random Forest (RF)
# *************************************************
set.seed(754)
rf_model = randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                                           Parch + Fare + Embarked + Title + 
                                           FamilySizeD + Child + Mother,
                        data = train)
plot(rf_model, ylim = c(0, 0.36))
legend("topright", colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

# Variable importance 
# Get importance
importance = importance(rf_model)
varImportance = data.frame(Variables = row.names(importance),
                           Importance = round(importance[, "MeanDecreaseGini"], 2))
# Create a rank variable based on importance
rankImportance = varImportance %>%
    mutate(Rank = paste0("#", dense_rank(desc(Importance))))

# relative importance of variable
ggplot(data = rankImportance,
       mapping = aes(x = reorder(Variables, Importance),
                     y = Importance,
                     fill = Importance)) +
    geom_bar(stat = "identity") +
    geom_text(mapping = aes(x = Variables, y = 0.5, label = Rank),
              hjust = 0, vjust = 0.55, size = 4, colour = "red") +
    labs(x = "Variable") +
    coord_flip() +
    theme_few()


# Prediction
prediction = predict(rf_model, test)

# Save the solution to a dataframe with two columns
solution = data.frame(PassengerID = test$PassengerId, 
                      Survived = prediction)
# write_csv(solution, file = "rf_mod_Solution.csv")

# *************************************************
# GBDT
# *************************************************




################################################################################
################################################################################
# 4. Training the Classification Models
######################################################################
####                  logistic regression                         ####
######################################################################
## Training the model
logit.model = glm(Survived ~ ., data = training, family = "binomial")
summary(logit.model)
logit.response = predict(logit.model, testing, type = "response")
logit.predict = ifelse(logit.response > 0.5, "1", "0")

## Results
### confusionMatrix
table(Predict = logit.predict, Survived = testing$Survived)
logit.accuracy = mean(logit.predict == testing$Survived)
logit.accuracy

# contrasts(Survived)

confusionM = confusionMatrix(logit.predict, testing$Survived)
confusionM

names(confusionM)
confusionM$positive
confusionM$table
confusionM$overall
confusionM$byClass
confusionM$dots

### ROC culer
library(pROC)
logitROC = roc(testing$Survived, logit.response,
               levels = levels(as.factor(testing$Survived)))
plot(logitROC, type = "S", print.thres = 0.5)
###############################################################################
### Model Evalulation - 10-folds CV
confusion = list(NA)
for(i in 1:5) {
    training1 = train[folds != i, ]
    testing1 = train[folds == i, ]
    logit.model1 = glm(Survived ~ ., data = training1, family = "binomial")
    logit.response1 = predict(logit.model1, testing1, type = "response")
    logit.predict1 = ifelse(logit.response1 > 0.5, "1", "0")
    confusion[[i]] = confusionMatrix(logit.predict1, testing1$Survived)
}
confusion
## Solution
my.prediction = predict(rpart.model, newdata = test, type = "class")
my_solution = data.frame(PassengerId = test$PassengerId, Survived = rpart.prediction)
nrow(my_solution)
write.csv(my_solution, file = "F:/Rworkd/my_solution/my_solution.csv",
          row.names = FALSE)

##########################################################################
##########################################################################
library(caret)
set.seed(1056)
Contr = trainControl(method = "repeatedcv", repeats = 5)
logistic.fit = train(Survived ~., data = training, 
                     method = "glm", 
                     trControl = Contr)
logistic.predict = predict(logistic.fit, newdata = testing, type = "response")


## Solution
my.prediction = predict(rpart.model, newdata = test, type = "class")
my_solution = data.frame(PassengerId = test$PassengerId, 
                         Survived = rpart.prediction)
nrow(my_solution)
write.csv(my_solution, file = "F:/Rworkd/my_solution/my_solution.csv",
          row.names = FALSE)

######################################################################
####        classification and regression trees                   ####
######################################################################
training_two$family_size <- train$SibSp + train$Parch + 1

library(rpart)
rpart.model = rpart(Survived ~ .,
                    data = training,
                    method = "class",
                    control = rpart.control(minsplit = 50, cp = 0.01)
                    
                    summary(rpart.model)
                    ## Visualizing the tree
                    ### method 1
                    library(rpart)
                    plot(rpart.model, compress = TRUE)
                    text(rpart.model, use.n = TRUE)
                    
                    ### method 2
                    library(rattle)
                    fancyRpartPlot(rpart.model)
                    
                    ### method 3
                    library(partykit)
                    rpart1a = as.party(rpart.model)
                    plot(rpart1a)
                    
                    rpart.predict = predict(rpart.model, testing, type = "class")
                    
                    ## table(rpart.predict, testing$Survived)
                    ## rpart.accuracy = mean(rpart.predict == testing$Survived)
                    ## rpart.accuracy
                    
                    confusionMatrix(rpart.predict, testing$Survived)
                    
                    # Create the ROC curve
                    library(pROC)
                    rpartROC = roc(testing$Survived, rpart.response,
                                   levels = levels(as.factor(testing$Survived)))
                    plot(rpartROC, type = "S", print.thres = 0.5)
                    
                    
                    ## Solution
                    my.prediction = predict(rpart.model, newdata = test, type = "class")
                    my_solution = data.frame(PassengerId = test$PassengerId, Survived = rpart.prediction)
                    nrow(my_solution)
                    write.csv(my_solution, file = "F:/Rworkd/my_solution/my_solution.csv",
                              row.names = FALSE)
                    
                    ###########################################################################
                    
                    cvCtrl = trainControl(method = "repeatdecv", repeats = 3, 
                                          summaryFunction = twoClassSummary, 
                                          classProbs = FALSE)
                    set.seed(1)
                    rpartTune = train(Survived ~ ., 
                                      data = training, 
                                      method = "rpart", 
                                      tuneLength = 30, 
                                      metric = "ROC", 
                                      trControl = cvCtrl)
                    rpartTune
                    plot(rpartTune, scales = list(x = list(log = 10)))
                    rpartPred = predict(rpartTune, testing)
                    confusionMatrix(rpartPred2, testing$Species)
                    rpartProbs = predict(rpartTune, testing, type = "prob")
                    head(rpartProbs)
                    ## Create the ROC curve
                    library(pROC)
                    rpartROC = roc(testing$Species, rpartProbs[, "PS"],
                                   levels = rev(testProbs$Class))
                    rpartROC
                    plot(rpartROC, type = "S", print.thres = 0.5)
                    ## Solution
                    my.prediction = predict(rpart.model, newdata = test, type = "class")
                    my_solution = data.frame(PassengerId = test$PassengerId, Survived = rpart.prediction)
                    nrow(my_solution)
                    write.csv(my_solution, file = "F:/Rworkd/my_solution/my_solution.csv",
                              row.names = FALSE)




