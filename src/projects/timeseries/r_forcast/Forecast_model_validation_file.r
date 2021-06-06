# LBShell Forecast testing file
# -----------------------------------------------------------------
# Anshu Gupta | 2/5/18
# -----------------------------------------------------------------
# Need to specify arguments within the user input section
# The file will return the forecast (for the specified methods)
# ----------------------------------------------------------------
rm(list = ls())

# Turning off warning 
options(warn = 0)
# to switch warning back
#options(warn = 0) 

setwd("\\\\home7\\DecisionScience\\Anshu Gupta\\LBShell Forecast\\")


# ---------------------------------------------------------------------
# Calling in libraries 
# ---------------------------------------------------------------------
library(RODBC)
library(dplyr)
library(foreach)
library(randomForest)
library(forecast)


# ---------------------------------------------------------------------
# Loading the function file 
# ---------------------------------------------------------------------
filepath = "\\\\home7\\Forecast\\LBShell_forecast\\Rcode\\"
source(paste(filepath, "Forecast_function_file.R", sep = ""))


# ---------------------------------------------------------------------
# Getting the forecast
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# User Input 
# ---------------------------------------------------------------------
Region = "ASIA"                                                                 # NA for North America | EU for Europe | ASIA for Asia
Type   = "MB"                                                                   # MB for MicroBulk | LB for Liquidbulk | ALL for both 
PrimaryTerminal = c("857")                                                      # Primary Terminal for which you need the result eg: ("'X14'") in asia. C81 in EU, 857 in NA or "ALL" 
# PrimaryTerminal = c("'857', '334', '389'")                                    # or use this format for multiple terminals 
model_pathname = "\\\\home7\\Forecast\\Asia_MB_Forecast\\models\\0401_0830\\"   # stored model pathname

# ---------------------------------------------------------------------
# Test Period
# ---------------------------------------------------------------------
Start_date = "2018-09-01" # Start Date (YYYY-MM-DD)
End_date   = "2018-10-30" # End Date (YYYY-MM-DD)

pred_horizon = 24 #hours

channelname = "LBShell" # ODBC channel name to connect to database

Method = c("ARIMA", "ETS", "NN", "RF","NNX", "ARIMAX") # Choose between "ARIMA", "ARIMAX", "ETS", "NN", "NNX", "RF"


# ---------------------------------------------------------------------
# Group forecast - parallel (saves time) - issues with getting it working| using sequential loop
# ---------------------------------------------------------------------
## Get list of locnum
input_path = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Input\\"
LN_Data = read.csv(file = paste0(input_path, "Asia_MB_Locnum_with_telemetry.csv"),
                   stringsAsFactors = F)
LN_Data = LN_Data %>% filter(DOL == 'T4')

#LN_Data = head(LN_Data, 30)

Forecast_Data = foreach(j = 1:dim(LN_Data)[1], .combine = 'rbind', .packages = c("RODBC", "dplyr", "forecast")) %do% {
  print(paste(Sys.time(), "Executing loop", j))
  Data = get_prediction_horizon_forecast(LN_Data$LocNum[j],Start_date, End_date,
                                         Region, Type,channelname, pred_horizon, model_pathname)
  #get_forecast_level(LN_Data$LocNum[j], Start_date, End_date, Region, Type, channelname, pathname, Method)
  #train_forecast_model(, Start_date, End_date, Region, Type, chan   nelname, method)
}

#stopCluster(cl)
outpath    = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\BestMethod\\"
systemtime = paste(Sys.Date(), strftime(Sys.time(), format = "%H-%M-%S"), sep = "_")
filename = paste0(outpath, paste(Start_date, End_date, sep = "_"))

write.csv(Forecast_Data, file = paste(filename, "forecast_data.csv",sep = "_"), row.names = F)

# ---------------------------------------------------------------------
# Computing MAPE for the forecasted method
# ---------------------------------------------------------------------
MAPE_Data = get_MAPE(Forecast_Data)
write.csv(MAPE_Data, file = paste(filename, "MAPE_data.csv",sep = "_"), row.names = F)

# ---------------------------------------------------------------------
# Best Methof file
# ---------------------------------------------------------------------
Output_Data = MAPE_Data %>%
  select(LocNum, best_method, minvalue, num_obs)
write.csv(Output_Data, file = paste(filename, "Best_Method_table.csv", sep = "_"), row.names = F)

