library(DMwR)
Analysis_txt = read.table("http://www.dcc.fc.up.pt/~ltorgo/DataMiningWithR/DataSets/Analysis.txt",
                       header = F,
                       dec = ".",
                       col.names = c("season", "size", "speed", 
                                     "mxPH", "mnO2", "Cl", "NO3", "NH4", "oPO4", "PO4", "Chla", 
                                     "a1", "a2", "a3", "a4", "a5", "a6", "a7"),
                       na.strings = "XXXXXXX")
Eval_txt = read.table("http://www.dcc.fc.up.pt/~ltorgo/DataMiningWithR/DataSets/Eval.txt",
                      header = FALSE,
                      dec = ".",
                      col.names = c(),
                      na.strings = "XXXXXXX")

Sols_txt = read.table("http://www.dcc.fc.up.pt/~ltorgo/DataMiningWithR/DataSets/Sols.txt",
                      header = FALSE,
                      dec = ".",
                      col.names = c(),
                      na.strings = "XXXXXXX")




# data distribution inforimation
# method 1
print(summary(algae))

# method 2
library(Hmisc)
print(describe(algae))
