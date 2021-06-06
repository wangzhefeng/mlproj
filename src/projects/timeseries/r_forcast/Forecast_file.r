# LBShell Forecast testing file
# -----------------------------------------------------------------
# Anshu Gupta | 2/5/18
# -----------------------------------------------------------------
# Need to specify arguments within the user input section
# The file will return the forecast (for the specified methods)
# -----------------------------------------------------------------
rm(list = ls())

# Turning off worning 
options(warn = 0)
# to switch warning back
# options(warn = 0) 

setwd("\\\\home7\\DecisionScience\\Anshu Gupta\\LBShell Forecast\\")


# -----------------------------------------------------------------
# Calling in libraries 
# -----------------------------------------------------------------
library(RODBC)
library(dplyr)
library(foreach)
library(randomForest)
library(forecast)


# -----------------------------------------------------------------
# Loading the function file 
# -----------------------------------------------------------------
filepath = "\\\\home7\\Forecast\\LBShell_Forecast\\Rcode\\"
source(paste(filepath, "Forecast_function_file.R", sep = ""))


# -----------------------------------------------------------------
# Getting the forecast
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# User Input 
# ----------------------------------------------------------------
Region = "ASIA"                                                                 # NA for North America | EU for Europe | ASIA for Asia
Type   = "MB"                                                                   # MB for MicroBulk | LB for Liquidbulk | ALL for both 
PrimaryTerminal = c("857")                                                      # Primary Terminal for which you need the result eg: ("'X14'") in asia. C81 in EU, 857 in NA or "ALL" 
# PrimaryTerminal = c("'857', '334', '389'")                                    # or use this format for multiple terminals 
model_pathname = "\\\\home7\\Forecast\\Asia_MB_Forecast\\models\\0401_0830\\"   # stored model pathname

# -----------------------------------------------------------------
# Test Period
# -----------------------------------------------------------------
Start_date = Sys.time()
End_date   = Start_date + 168*60*60 # 24 hr in the future 

# -----------------------------------------------------------------
# period over which validation was ran
# -----------------------------------------------------------------
validation_start_date = "2018-09-01" # Start Date (YYYY-MM-DD)
validation_end_date   = "2018-10-30" # End Date (YYYY-MM-DD)

channelname = "LBShell" # ODBC channel name to connect to database


# -----------------------------------------------------------------
## Get list of locnum
input_path = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\BestMethod\\"

bestmethod_filename = paste0(input_path, paste(validation_start_date, validation_end_date, sep = "_"))
bestmethod_dta = read.csv(file = paste(bestmethod_filename, "Best_Method_table.csv", sep = "_"), stringsAsFactors = F)
#bestmethod_dta = bestmethod_dta[1:5, ]

# bestmethod_dta = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\BestMethod\\2018-09-01_2018-10-30_Best_Method_table.csv"


# -----------------------------------------------------------------
runout_data = foreach(j = 1:dim(bestmethod_dta)[1], 
                      .combine = 'rbind', 
                      .packages = c("RODBC", "dplyr", "forecast")) %do% {
  LocNum = bestmethod_dta$LocNum[j]
  print(paste(Sys.time(), LocNum, "Executing loop", j))
  runout_forecast(LocNum, Start_date, End_date, Region, Type, channelname, model_pathname, bestmethod_filename)
}

runout_data$runout_time_prediction = as.POSIXct(runout_data$runout_time_prediction, origin = '1970-01-01')


# -----------------------------------------------------------------
# 输出结果
# -----------------------------------------------------------------
outpath = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\RunOutTime\\"
starttime = paste(as.Date(Start_date), strftime(Start_date, format = "%H-%M-%S"), sep = "_")
write.csv(runout_data, file = paste0(outpath, starttime, "_runout_data.csv"), row.names = F)

# ----------------------------------------------------------------------------------
# Forecast usage rate data
# filepath = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\UsageRate\\IndividualUsageRate\\"
# forecast_data = read.csv(file = paste0(filepath, bestmethod_dta$LocNum[1], "_usage_rate.csv"), stringsAsFactors = F)
# names(forecast_data) = c("Time", paste0(bestmethod_dta$LocNum[1], "_usage_rate"))
# 
# for (i in 2:dim(bestmethod_dta)[1]){
#   LocNum = bestmethod_dta$LocNum[i]
#   print(paste(Sys.time(), LocNum, "Executing loop", i))
#   forecast_dta = read.csv(file = paste0(filepath, LocNum, "_usage_rate.csv"), stringsAsFactors = F)
#   names(forecast_dta) = c("Time", paste0(LocNum, "_usage_rate"))
#   forecast_data = inner_join(forecast_data, forecast_dta, by = "Time")
# }
# 
# filepath = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\UsageRate\\"
# write.csv(forecast_data, file = paste0(filepath, starttime, "_forecasted_usage_rate.csv"), row.names = F)
