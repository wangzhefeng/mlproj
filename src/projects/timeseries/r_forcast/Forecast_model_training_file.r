# Forecast training file
# -----------------------------------------------------------------
# Anshu Gupta | 2/5/18
# -----------------------------------------------------------------
# Need to specify arguments within the user input section
# The forecast models will be trained for specified methods and 
# stored in the folder provided as the pathname

# ----------------------------------------------------------------
rm(list = ls())
setwd("\\\\home7\\DecisionScience\\Anshu Gupta\\Asia MB forecast")


# -----------------------------------------------------------------------------------------------------
# Calling in libraries (make sure you have them installed)
# -----------------------------------------------------------------------------------------------------
library(RODBC)
library(tidyverse)
library(foreach)
library(doParallel)
library(randomForest)
library(forecast)


# -----------------------------------------------------------------------------------------------------
# User Input 
# -----------------------------------------------------------------------------------------------------
Region = "ASIA"                                                                 # NA for North America | EU for Europe | ASIA for Asia
Type   = "MB"                                                                   # MB for MicroBulk | LB for Liquidbulk | ALL for both 
PrimaryTerminal = c("MT")                                                       # Primary Terminal for which you need the result eg: ("'X14'") in asia. C81 in EU, 857 in NA or "ALL" 
model_pathname = "\\\\home7\\Forecast\\Asia_MB_Forecast\\models\\0401_0830\\"   # model storage pathname 
# PrimaryTerminal = c("'857', '334', '389'")                                    # or use this format for multiple terminals 


# -----------------------------------------------------------------------------------------------------
# Model Training Period
# -----------------------------------------------------------------------------------------------------
Start_date = "2018-04-01"                                                       # Start Date (YYYY-MM-DD)
End_date   = "2018-08-30"                                                       # End Date (YYYY-MM-DD)
channelname = "LBShell"                                                         # ODBC channel name to connect to database (server name: LRPSQP05\LRPSQP05)
Method = c("ARIMA", "ETS", "NN", "RF","NNX", "ARIMAX")                          # Choose between "ARIMA", "ARIMAX", "ETS", "NN", "NNX", "RF"
#Method = c("ARIMA", "ETS", "NN", "RF")
#Method = "RF"


# -----------------------------------------------------------------------------------------------------
# Calling the forecast funtion file
# -----------------------------------------------------------------------------------------------------
filepath = "\\\\home7\\Forecast\\LBShell_Forecast\\Rcode\\"
source(paste(filepath, "Forecast_function_file.R", sep = ""))


#------------------------------------------------------------------------------------------------------
## Get list of locnum (either for the terminal or read in from a file)
# -----------------------------------------------------------------------------------------------------
input_path = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Input\\"
LN_Data = read.csv(file = paste0(input_path, "Asia_MB_Locnum_with_telemetry.csv"), stringsAsFactors = F)
LN_Data = LN_Data %>% filter(DOL == 'T4')
#LN_Data = head(LN_Data, 7)


cl = makeCluster(detectCores() - 1,                                             # Creating cluster for parallel execution / can change number of dedicated cores here
                 outfile = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\Asia_log_all_V3.txt")    # log file

registerDoParallel(cl)

foreach(j = 1:dim(LN_Data)[1], 
        .packages = c("RODBC", "tidyverse", "forecast")) %dopar% {
  print(paste(Sys.time(), "Executing loop", j))
  train_forecast_model(LN_Data$LocNum[j], 
                       Start_date, 
                       End_date, 
                       Region, 
                       Type, 
                       channelname, 
                       model_pathname, 
                       Method, 
                       driverreadingflag)
}

stopCluster(cl)


