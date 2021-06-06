# Function file 
# Contains all the function for training and testing forecast model 
# ------------------------------------------------------------------
# -------------------------------------------------------------------
# Assumptions:
# 1. The linear usage outlier (mu+3sd) are substituted for meadian value
# 2. The cust pattern with mixed reading for zero and non-zero gets all
#    the values linearized over non-zero time frame
# ---------------------------------------------------------------------
# Anshu Gupta
# 2/5/18
# Buch of chnages after that...

# --------------------------------------------------------------------
# Defining functions 
# ------------------------------------------------------------------

get_region = function(Region){
  if (Region == "NA"){
    Regionname = c("'NOR', 'SOU', 'WEST' ")
    DB = "[NA_LBLogist_Rpt]"
  } else if (Region == "EU") {
    Regionname = c("'UK', 'CN', 'CE', 'IB'")
    DB = "[EU_LBLogist_Rpt]"
  } else if (Region == "ASIA") {
    Regionname = c("'KR', 'SE', 'TW', 'XN'")
    DB = "[EU_LBLogist_Rpt]"
  } else {
    print("Supply correct region name") 
    print("NA for North America") 
    print("EU for Europe")
    print("ASIA for Asia")
    print("Using default - North America")
    Regionname = c("'NOR', 'SOU', 'WEST' ")
    DB = "[NA_LBLogist_Rpt]"
  }
  return(Regionname)
}

get_database_name = function(Region){
  if (Region == "NA"){
    #Regionname = c("'NOR', 'SOU', 'WEST' ")
    DB = "[NA_LBLogist_Rpt]"
  } else if (Region == "EU") {
    #Regionname = c("'UK', 'CN', 'CE', 'IB'")
    DB = "[EU_LBLogist_Rpt]"
  } else if (Region == "ASIA") {
    #Regionname = c("'KR', 'SE', 'TW', 'XN'")
    DB = "[EU_LBLogist_Rpt]"
  } else {
    print("Supply correct region name") 
    print("NA for North America") 
    print("EU for Europe")
    print("ASIA for Asia")
    print("Using default - North America")
    # Regionname = c("'NOR', 'SOU', 'WEST' ")
    DB = "[NA_LBLogist_Rpt]"
  }
  return(DB)
}

get_type = function(Type){
  if (Type == "LB"){          TerminalType = 2
  } else if (Type == "MB") {  TerminalType = 1
  } else if (Type == "ALL") { TerminalType = c("1, 2")
  } else {
    print("Supply correct Type") 
    print("MB for MicroBulk") 
    print("LB for LiquidBulk")
    print("ALL for Both")
    print("Using default - MB for Microbulk")
    TerminalType = 1
  }
  return(TerminalType)
}


get_customer_profile = function(LocNum, Region, Type, channelname)  {
  
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  
  # --------------------------------------------------------------------
  channel = RODBC::odbcConnect(channelname)
  # Getting the customer profile data 
  CP_sql = paste("SELECT AccountNum
                 ,[LocNum]
                 ,[ProductAbbrev]
                 ,[DemandType]
                 ,[DlvryStatus]
                 ,[ProductClass]
                 ,[PrimaryTerminal]
                 ,[FullLoadInch]
                 ,[FullLoadGals]
                 ,[RunoutInch]
                 ,[RunoutGals]
                 ,[FullTrycockInches]
                 ,[FullTrycockGals]
                 ,[GalsPerInch]
                 ,[LinearOpPattern]
                 ,[EstDlvryDate]
                 ,[OpMonShift1]
                 ,[OpMonShift2]
                 ,[OpMonShift3]
                 ,[OpTueShift1]
                 ,[OpTueShift2]
                 ,[OpTueShift3]
                 ,[OpWedShift1]
                 ,[OpWedShift2]
                 ,[OpWedShift3]
                 ,[OpThuShift1]
                 ,[OpThuShift2]
                 ,[OpThuShift3]
                 ,[OpFriShift1]
                 ,[OpFriShift2]
                 ,[OpFriShift3]
                 ,[OpSatShift1]
                 ,[OpSatShift2]
                 ,[OpSatShift3]
                 ,[OpSunShift1]
                 ,[OpSunShift2]
                 ,[OpSunShift3]
                 ,[OpMonShift1StartTime]
                 ,[OpMonShift1EndTime]
                 ,[OpMonShift2StartTime]
                 ,[OpMonShift2EndTime]
                 ,[OpMonShift3StartTime]
                 ,[OpMonShift3EndTime]
                 ,[OpTueShift1StartTime]
                 ,[OpTueShift1EndTime]
                 ,[OpTueShift2StartTime]
                 ,[OpTueShift2EndTime]
                 ,[OpTueShift3StartTime]
                 ,[OpTueShift3EndTime]
                 ,[OpWedShift1StartTime]
                 ,[OpWedShift1EndTime]
                 ,[OpWedShift2StartTime]
                 ,[OpWedShift2EndTime]
                 ,[OpWedShift3StartTime]
                 ,[OpWedShift3EndTime]
                 ,[OpThuShift1StartTime]
                 ,[OpThuShift1EndTime]
                 ,[OpThuShift2StartTime]
                 ,[OpThuShift2EndTime]
                 ,[OpThuShift3StartTime]
                 ,[OpThuShift3EndTime]
                 ,[OpFriShift1StartTime]
                 ,[OpFriShift1EndTime]
                 ,[OpFriShift2StartTime]
                 ,[OpFriShift2EndTime]
                 ,[OpFriShift3StartTime]
                 ,[OpFriShift3EndTime]
                 ,[OpSatShift1StartTime]
                 ,[OpSatShift1EndTime]
                 ,[OpSatShift2StartTime]
                 ,[OpSatShift2EndTime]
                 ,[OpSatShift3StartTime]
                 ,[OpSatShift3EndTime]
                 ,[OpSunShift1StartTime]
                 ,[OpSunShift1EndTime]
                 ,[OpSunShift2StartTime]
                 ,[OpSunShift2EndTime]
                 ,[OpSunShift3StartTime]
                 ,[OpSunShift3EndTime]
                 
                 FROM ", DB,".[dbo].[CustomerProfile] 
                 WHERE Locnum = '", LocNum, "'")
  
  # CP_sql = readLines("Asia_LB_CustomerProfile_V2.txt", warn = F)
  # CP_sql1 = paste(paste(CP_sql[CP_sql!='--'],collapse = ' '), "WHERE Locnum = '", LocNum, "'")
  CP_Data = RODBC::sqlQuery(channel, CP_sql)
  
  RODBC::odbcClose(channel)
  
  
  # -----------------------------------------------------------------------------------
  # Creating an operating pattern
  # -----------------------------------------------------------------------------------
  
  #Data_CP = CP_Data[CP_Data$LocNum%in%Locnum,]
  Data    = CP_Data
  
  TimeData  = list()  # Variable to store hourly time element
  ValueData = list() # Variable to store hourly usage element
  Reading   = numeric()
  
  ind_V1 = which(names(Data) == "OpMonShift1")
  ind_D1 = which(names(Data) == "OpMonShift1StartTime")
  
  for (j in 1:21){
    V1 = as.numeric(Data[ind_V1 + (j-1)])
    D1 = as.POSIXct(as.numeric(Data[ind_D1 + 2*(j-1)]), origin =  "1970-01-01")
    D2 = as.POSIXct(as.numeric(Data[ind_D1 + 1 + 2*(j-1)]), origin =  "1970-01-01")
    Seq = seq(D1, D2, by = "hour")
    start_bias = as.numeric(difftime(D1, as.POSIXct("1970-01-01"), units = "hours"))
    end_bias   = as.numeric(difftime(as.POSIXct("1970-01-02"), D2, units ="hours"))
    #ValueData[j] = list(c(rep(0, start_bias), rep(V1, length(Seq) - 1)))
    if (start_bias >= 0) {
      if (end_bias >= 1) {
        ValueData[j] = list(c(rep(0, start_bias), 
                              rep(V1, length(Seq)),
                              rep(0, end_bias - 1)))
      } else {
        ValueData[j] = list(c(rep(0, start_bias), 
                              rep(V1, length(Seq) - 1),
                              rep(0, end_bias)))
      }
    }
    
  }
  
  x = seq(1,21,3)
  for (v in x){
    VData = rep(0,24)
    
    ind_v1 = unlist(ValueData[v]) != 0
    ind_v2 = unlist(ValueData[v + 1]) != 0
    ind_v3 = unlist(ValueData[v + 2]) != 0
    
    VData[ind_v1] = unlist(ValueData[v])[ind_v1]
    VData[ind_v2] = unlist(ValueData[v + 1])[ind_v2]
    VData[ind_v3] = unlist(ValueData[v + 2])[ind_v3]
    
    Reading = c(Reading,VData)
    
    # # VData = c(unlist(ValueData[v]),unlist(ValueData[v+1]),unlist(ValueData[v+2]))
    #  if (length(VData) != 24){
    #    VData = c(VData, rep(0, 24 - length(VData)))
    #  }
    
  }
  
  
  if (length(Reading) > 168){
    print("Reading not 168")
    break}
  
  # Choosing a dummy Monday to create a sequence of days and time for a week
  dummy_week_seq = seq(as.POSIXct("2017-12-18"), by = "hour", length.out = 168)
  weekdays = weekdays(dummy_week_seq)
  hours = lubridate::hour(dummy_week_seq)
  
  CusProfile_Reading = cbind.data.frame(weekdays, hours, Reading)
  return(CusProfile_Reading)
}


get_usage_rate_data = function(LocNum, Start_date, End_date, Region, Type, 
                               channelname, driverreadingflag = 1, remvarianceflag = 0)  {
  
  # packages 
  require(RODBC)
  require(tidyverse)
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  # --------------------------------------------------------------------
  #channel = RODBC::odbcConnect("LBShell")
  channel = RODBC::odbcConnect(channelname)
  # Getting the customer profile data 
  CP_sql  = CP_sql = paste("SELECT [LocNum]
                           ,[ProductAbbrev]
                           ,[GalsPerInch]
                           FROM ", DB,".[dbo].[CustomerProfile] 
                           WHERE Locnum = '", LocNum, "'")
  #CP_sql1 = paste(paste(CP_sql[CP_sql!='--'],collapse = ' '), "WHERE Locnum = '", LocNum, "'")
  CP_Data = RODBC::sqlQuery(channel, CP_sql)
  
  # Telemetry Reading Data
  # Reading data from now to 6 months back
  # TR_end_time   = Sys.time() # Current time 
  # TR_start_time = seq(as.Date(TR_end_time), length = 2, by = "-6 months")[2] # 6 months back from today
  # TR_start_time = as.POSIXct(TR_start_time)
  TR_start_time = as.POSIXct(Start_date, origin = "1970-01-01")
  TR_end_time   = as.POSIXct(End_date, origin = "1970-01-01")
  
  TR_sql = paste("SELECT R.[AccountNum]
                 ,R.[LocNum]
                 ,[ReadingDate]
                 ,CASE WHEN [ReadingUOM] = 1 THEN [ReadingLevel] / [GalsPerInch] ELSE [ReadingLevel] END AS ReadingLevel
                 ,[ReadingType]
                 FROM", DB,".[dbo].[Readings] R
                 JOIN", DB,".[dbo].[CustomerProfile] CP
                 ON R.LocNum = CP.LocNum
                 WHERE ReadingDate >= '", TR_start_time, "'",
                 "AND ReadingDate <'", TR_end_time, "'",
                 "AND (ReadingLevel > 0 or (readinglevel = 0 and readingtype = 2))
                 AND ReadingUOM != 2
                 AND R.Locnum = '", LocNum, "'",
                 "AND ReadingStatus != 2
                 ORDER BY ReadingDate")
  
  TR_Data = RODBC::sqlQuery(channel, TR_sql, as.is = T)
  
  RODBC::odbcClose(channel)
  
  
  
  # ------------------------------------------------------------------------- 
  # Running only if at least 5 telemetry data exist for the customer
  if(dim(TR_Data)[1] > 5) {
    # Adding reading in gals
    TR_Data$ReadingDate = as.POSIXct(TR_Data$ReadingDate, tz = 'EST')
    TR_Data$ReadingLevel = as.numeric(TR_Data$ReadingLevel)
    TR_Data$Reading_Gals = TR_Data$ReadingLevel*CP_Data$GalsPerInch
   
    # Added to remove delivery data
    if (driverreadingflag == 0){
      TR_Data = TR_Data %>% filter(ReadingType != 2, ReadingType != 3)
    }
    
    
    
    # Getting the customer profile data
    CusProfile_master_Reading = get_customer_profile(LocNum, Region, Type, channelname)
    names(CusProfile_master_Reading) = c("weekdays", "hours", "Value")
    CusProfile_history        = get_cusprofile_history(LocNum, Start_date, End_date, Region, Type, channelname)
    
    CusProfile_Reading = CusProfile_history %>%
      mutate(weekdays = weekdays(Time),
             hours    = lubridate::hour(Time)) %>%
      left_join(CusProfile_master_Reading, by = c("weekdays", "hours")) %>%
      mutate(Reading  = ifelse(is.na(Reading) == 1, Value, Reading )) %>%
      select(Time, Reading)
    CusProfile_Reading$Time =  as.POSIXct(as.character((CusProfile_Reading$Time)), tz = "EST")
    # attr(CusProfile_Reading$Time, 'tzone') = 'EST'
    
    # Creating the time sequence for the length of dataset 
    data_length_hours = round(as.numeric(difftime(TR_end_time, TR_start_time, units = "hours")))
    Time_Seq = seq(TR_start_time, by = "hour", length.out = data_length_hours)
    Time_Seq = as.POSIXct(as.character((Time_Seq)), tz = "EST")
    # attr(Time_Seq, 'tzone') = 'EST'
    
    #Reading = head(rep(Reading, 105), 17545)
    Time_Seq = as.data.frame(Time_Seq)
    names(Time_Seq) = "Time"
    
    CusProfile = Time_Seq %>%
      mutate(weekdays = weekdays(Time),
             hours    = lubridate::hour(Time),
             LocNum   = LocNum)     %>%
      left_join(CusProfile_Reading, by = c("Time"))
    
    
    # ------------------------------------------------------------------------------
    # Telemetry Reading Data
    # ------------------------------------------------------------------------------
    
    TR_Data = TR_Data[order(TR_Data$ReadingDate), ]
    LinearApprox = approx(TR_Data$ReadingDate,
                          TR_Data$Reading_Gals,
                          CusProfile$Time, method = "linear", rule = 2)
    
    CusProfile$LinearReadings = LinearApprox$y
    CusProfile$LinearUsage = round(-c(0,diff(CusProfile$LinearReadings)),3)
    CusProfile$LinearUsage[CusProfile$LinearUsage < 0] = NA
    
    #CusProfile$LinearUsage[is.na(CusProfile$LinearUsage) == 1] = median(CusProfile$LinearUsage, na.rm = T)
    CusProfile$LinearUsage = zoo::na.locf(CusProfile$LinearUsage)
    # Removing outliers | replacing the value with mean (double pass)
    n.pass = 1
    for (i in 1:n.pass) {
      ind_nonzero = CusProfile$LinearUsage > 0
      HighVal = mean(CusProfile$LinearUsage[ind_nonzero]) + 3*sd(CusProfile$LinearUsage[ind_nonzero])
      CusProfile$LinearUsage[CusProfile$LinearUsage > HighVal] = median(CusProfile$LinearUsage, na.rm = T)
    }
    
    
    # Converting hourly readings to UsageRate (Gals/Hr) 
    # Using the customer profile data between each reading
    
    Reading_Time = unique(as.POSIXct(TR_Data$ReadingDate), na.rm = T)
    Reading_Time = Reading_Time[is.na(Reading_Time) == 0]
    for (i in 2:length(Reading_Time)){
      ind_BR = (CusProfile$Time > Reading_Time[i-1] & CusProfile$Time <= Reading_Time[i])
      Num_Readings = sum(CusProfile$Reading[ind_BR])
      if ((Num_Readings/100 == sum(ind_BR)) | (Num_Readings == 0)) {
        CusProfile$UsageRate_Gals_per_Hr[ind_BR] = CusProfile$LinearUsage[ind_BR]
      } else {
        CusProfile$UsageRate_Gals_per_Hr[ind_BR] = (sum(CusProfile$LinearUsage[ind_BR]))*
          (CusProfile$Reading[ind_BR]/Num_Readings)
      }
    }
    ind_na = is.na(CusProfile$UsageRate_Gals_per_Hr)
    CusProfile$UsageRate_Gals_per_Hr[ind_na] = CusProfile$LinearUsage[ind_na]
    CusProfile$UsageRate_Gals_per_Hr = round(CusProfile$UsageRate_Gals_per_Hr, 3)

    
    # Removing high and low outliers from the data based on hour of the day
    num_week = unique(CusProfile$weekdays)
    num_hours = unique(CusProfile$hours)
    if (remvarianceflag == 1){
      for (w in 1:length(num_week)){
        for (h in 1:length(num_hours)){
          dta = CusProfile %>% select(Time, weekdays, hours, UsageRate_Gals_per_Hr) %>%
            filter(hours == num_hours[h], weekdays == num_week[w])
          highval = median(dta$UsageRate_Gals_per_Hr) + 3*sd(dta$UsageRate_Gals_per_Hr)
          lowval  = median(dta$UsageRate_Gals_per_Hr) - 3*sd(dta$UsageRate_Gals_per_Hr)
          dta$UsageRate_Gals_per_Hr[dta$UsageRate_Gals_per_Hr > highval] = median(dta$UsageRate_Gals_per_Hr)
          dta$UsageRate_Gals_per_Hr[dta$UsageRate_Gals_per_Hr < lowval]  = median(dta$UsageRate_Gals_per_Hr)
          CusProfile$UsageRate_Gals_per_Hr[CusProfile$Time%in%dta$Time] = dta$UsageRate_Gals_per_Hr
        }
      }
    }
    
    
    CusProfile = CusProfile %>%
      select(LocNum, Time, Reading, LinearReadings, LinearUsage, UsageRate_Gals_per_Hr)
    names(CusProfile) = c("LocNum", "Time", "CustomerPattern", "LinearUsage_Gals",
                          "LinearUsage_Gals_per_Hr", "UsageRate_Gals_per_Hr")
    return(CusProfile)
    
  } else {
    # print(paste("No Telemetry data for LocNum = ", LocNum))
    # break
    CusProfile = data.frame(LocNum)
    CusProfile = CusProfile %>%
      mutate(Time = NA,
             CustomerPattern         = NA,
             LinearUsage_Gals        = NA,
             LinearUsage_Gals_per_Hr = NA,
             UsageRate_Gals_per_Hr   = NA)
  }
} 



prepare_training_data_shorthorizon = function(LocNum, Start_date, End_date, Region, 
                                              Type, channelname, driverreadingflag, remvarianceflag){
  
  
  
  # This function includes short term factors such as last 6hr/ 24hrs usage
  # use to train model when prediction horizon are short
  
    Data1 = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname, driverreadingflag)
 
  
  if(Data1 != 101 & dim(Data1)[1] > 504) {
    
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 6
    ind_S2 = Data1$Hours >= 7 & Data1$Hours <= 12
    ind_S3 = Data1$Hours >= 13 & Data1$Hours <= 18
    ind_S4 = Data1$Hours >= 19 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    Data1$Shift[ind_S4] = "Shift4"
    
    # Adding past 3 week usage information 
    Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data1 = Data1 %>% group_by(Week_Label) %>%
      mutate(WeekSum = sum(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data_avgWeek = aggregate(UsageRate_Gals_per_Hr ~ Week_Label, data = Data1, sum, na.rm = T)
    # colnames(Data_avgWeek)[2] = "WeekSum"
    # ind_avgweek = match(Data1$Week_Label, Data_avgWeek$Week_Label)
    # Data1$WeekSum = Data_avgWeek$WeekSum[ind_avgweek]
    
    cl = length(Data1$UsageRate_Gals_per_Hr)
    LastWeekEnd = cl-168
    Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    
    Week2St = 2*168+1
    Week2End = cl-2*168
    Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    
    Week3St = 3*168+1
    Week3End = cl-3*168
    Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    
    # Average Shift usage rate
    Data1 = Data1 %>% group_by(Shift) %>%
      mutate(Avg_Shift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Shift, Week_Label) %>%
      mutate(Avg_WeeklyShift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data_Shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, data = Data1, mean, na.rm = T)
    # colnames(Data_Shift)[2] = "Avg_Shift_Usage"
    # ind_Shift = match(Data1$Shift, Data_Shift$Shift)
    # Data1$Avg_Shift_Usage = Data_Shift$Avg_Shift_Usage[ind_Shift]
    
    # Average Daily usage rate
    Data1 = Data1 %>% group_by(Days, Week_Label) %>%
      mutate(Avg_WeeklyDay_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days) %>%
      mutate(Avg_Day_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days, Hours, Month) %>%
      mutate(Avg_Hourly_Usage_perMonth = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days, Hours) %>%
      mutate(Avg_Hourly_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1$usage6hrs = NA
    for (p in 7:dim(Data1)[1]){
      T1 = Data1$Time[p-6]
      T2 = Data1$Time[p]
      Data1$usage6hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    }
    
    Data1$usage24hrs = NA
    for (p in 25:dim(Data1)[1]){
      T1 = Data1$Time[p-24]
      T2 = Data1$Time[p]
      Data1$usage24hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    }
    
    Data1$usage60hrs = NA
    for (p in 61:dim(Data1)[1]){
      T1 = Data1$Time[p-60]
      T2 = Data1$Time[p]
      Data1$usage60hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    }
    
    # Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    # colnames(Data_Day)[2] = "Avg_Day_Usage"
    # ind_Day = match(Data1$Days, Data_Day$Days)
    # Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    # 
    # # Average Monthly usage rate
    # Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    # colnames(Data_Month)[2] = "Avg_Month_Usage"
    # ind_month = match(Data1$Month, Data_Month$Month)
    # Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    # 
    # 
    
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    #if (training == 0){
    # ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    # Data_F = Data1[ind_test, ]
      ind_na = is.na(Data1$Last3WeekSum)
      Data_F = Data1[!ind_na, ]
    
    Data_F = Data_F %>% select( UsageRate_Gals_per_Hr, Days, Shift, Hours, Month, 
                               LastWeekSum,Last2WeekSum, Last3WeekSum,
                               Avg_Shift_Usage, Avg_WeeklyShift_Usage, Avg_WeeklyDay_Usage,
                               Avg_Day_Usage, Avg_Hourly_Usage_perMonth, Avg_Hourly_Usage, 
                               usage6hrs, usage24hrs, usage60hrs)
    
    # } else  {
    
    # }
    
    # Data_F = dplyr::select(Data_F, Time, UsageRate_Gals_per_Hr, Days, Shift, Hours,
    #                        Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
    #                        Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3, 4)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    
  } else return(101)
  
  # # --------------------------------------------------------------------------
  # # --------------------------------------------------------------------------
  # # Hourly Prediction 
  # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
  # Xvar_Data = Data_F
  # 
  # if (methodclass == 1){ return(Timeseries_Data)}
  # else if (methodclass == 2) { return(Xvar_Data)}
  # else print("Provide a method class")
}


prepare_training_data_longhorizon = function(LocNum, Start_date, End_date, Region, Type, channelname){
  
  
  
  # This function includes long term factors such as hourly and weekly pattern
  # use to train model when prediction horizons are long
  
  Data1 = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname, 0)
  
  if(Data1 != 101) {
    if (dim(Data1)[1] > 504) {
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 6
    ind_S2 = Data1$Hours >= 7 & Data1$Hours <= 12
    ind_S3 = Data1$Hours >= 13 & Data1$Hours <= 18
    ind_S4 = Data1$Hours >= 19 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    Data1$Shift[ind_S4] = "Shift4"
    
    # Adding past 3 week usage information 
    Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data1 = Data1 %>% group_by(Week_Label) %>%
      mutate(WeekSum = sum(UsageRate_Gals_per_Hr, na.rm = T))
    
    cl = length(Data1$UsageRate_Gals_per_Hr)
    LastWeekEnd = cl-168
    Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    
    Week2St = 2*168+1
    Week2End = cl-2*168
    Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    
    Week3St = 3*168+1
    Week3End = cl-3*168
    Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    
    # Average Shift usage rate
    Data1 = Data1 %>% group_by(Shift) %>%
      mutate(Avg_Shift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Shift, Week_Label) %>%
      mutate(Avg_WeeklyShift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data_Shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, data = Data1, mean, na.rm = T)
    # colnames(Data_Shift)[2] = "Avg_Shift_Usage"
    # ind_Shift = match(Data1$Shift, Data_Shift$Shift)
    # Data1$Avg_Shift_Usage = Data_Shift$Avg_Shift_Usage[ind_Shift]
    
    # Average Daily usage rate
    Data1 = Data1 %>% group_by(Days, Week_Label) %>%
      mutate(Avg_WeeklyDay_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days) %>%
      mutate(Avg_Day_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days, Hours, Month) %>%
      mutate(Avg_Hourly_Usage_perMonth = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data1 = Data1 %>% group_by(Days, Hours) %>%
      mutate(Avg_Hourly_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data1$usage6hrs = NA
    # for (p in 7:dim(Data1)[1]){
    #   T1 = Data1$Time[p-6]
    #   T2 = Data1$Time[p]
    #   Data1$usage6hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data1$usage24hrs = NA
    # for (p in 25:dim(Data1)[1]){
    #   T1 = Data1$Time[p-24]
    #   T2 = Data1$Time[p]
    #   Data1$usage24hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data1$usage60hrs = NA
    # for (p in 61:dim(Data1)[1]){
    #   T1 = Data1$Time[p-60]
    #   T2 = Data1$Time[p]
    #   Data1$usage60hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    # colnames(Data_Day)[2] = "Avg_Day_Usage"
    # ind_Day = match(Data1$Days, Data_Day$Days)
    # Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    # 
    # # Average Monthly usage rate
    # Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    # colnames(Data_Month)[2] = "Avg_Month_Usage"
    # ind_month = match(Data1$Month, Data_Month$Month)
    # Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    # 
    # 
    
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    #if (training == 0){
    # ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    # Data_F = Data1[ind_test, ]
    ind_na = is.na(Data1$Last3WeekSum)
    Data_F = Data1[!ind_na, ]
    
    Data_F = Data_F %>% select( UsageRate_Gals_per_Hr, Days, Shift, Hours, Month, 
                                LastWeekSum,Last2WeekSum, Last3WeekSum,
                                Avg_Shift_Usage, Avg_WeeklyShift_Usage, Avg_WeeklyDay_Usage,
                                Avg_Day_Usage, Avg_Hourly_Usage_perMonth, Avg_Hourly_Usage)
                                
    # } else  {
    
    # }
    
    # Data_F = dplyr::select(Data_F, Time, UsageRate_Gals_per_Hr, Days, Shift, Hours,
    #                        Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
    #                        Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3, 4)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    
     } else return(101)
    } else return(101)
  
  # # --------------------------------------------------------------------------
  # # --------------------------------------------------------------------------
  # # Hourly Prediction 
  # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
  # Xvar_Data = Data_F
  # 
  # if (methodclass == 1){ return(Timeseries_Data)}
  # else if (methodclass == 2) { return(Xvar_Data)}
  # else print("Provide a method class")
}



train_forecast_model = function(LocNum, Start_date, End_date, Region, Type, channelname, 
                                pathname, Method, driverreadingflag, remvarianceflag){
  # train and saves a model for a given method 
  
  # if (method == "ARIMA" | method == "ETS" | method == "NN"){
  #   methodclass = 1 # timeseries data
  # } else methodclass = 2 #x variable data 
  
  #forecast_data = prepare_training_data(LocNum, Start_date, End_date, Region, Type, channelname)
  forecast_data = prepare_training_data_shorthorizon(LocNum, Start_date, End_date, Region, Type, channelname, driverreadingflag, remvarianceflag)
  #pathname = "\\\\home7\\DecisionScience\\Anshu Gupta\\LBShell Forecast\\model\\"
  for (iM in 1:length(Method)) {
    method = Method[iM]
  
  
    if (forecast_data == 101){
      model_fit = paste("No historical data for", LocNum, "between", Start_date, End_date)
    } else {
      
      if (method == "ARIMA")  {
        # Auto ARIMA
        model_fit      = forecast::auto.arima(ts(forecast_data[, 1]))
        filename       = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
      } else if (method == "ARIMAX")  {
        # ARX
        xVars           = data.matrix(forecast_data[, c(2: dim(forecast_data)[2])])
        model_fit       = forecast::auto.arima(ts(forecast_data[, 1]), xreg = xVars)
        filename        = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
      } else if (method == "ETS") {
        # Exponential smooting
        model_fit       = forecast::ets(ts(forecast_data[, 1]))
        filename        = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
      } else if (method == "NN") {
        # Neural network
        model_fit       = forecast::nnetar(ts(forecast_data[, 1]))
        filename        = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
      } else if (method == "NNX") {
        # Neural network with x variable 
        xVars           = data.matrix(forecast_data[, c(2: dim(forecast_data)[2])])
        model_fit       = forecast::nnetar(ts(forecast_data[, 1]), xreg = xVars, MaxNWts = 3000)
        filename        = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
      } else if (method == "RF") {
        # Random Forest
        set.seed(236)
        #if (dim(forecast_data)[2] > 11) {
        model_fit = randomForest::randomForest(formula = UsageRate_Gals_per_Hr ~ Days + Hours + Month  + Shift +
                                                 LastWeekSum + Last2WeekSum + Last3WeekSum  + Avg_Shift_Usage +
                                                 Avg_Day_Usage  + Avg_WeeklyShift_Usage + Avg_WeeklyDay_Usage +
                                                 Avg_Hourly_Usage_perMonth + Avg_Hourly_Usage,
                                               data = forecast_data,
                                               Importance = TRUE,
                                               ntree = 400)
      
        filename = paste(pathname, method, "_LH_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
        
        #forecast_data_SH = prepare_training_data_shorthorizon(LocNum, Start_date, End_date, Region, Type, channelname)
        model_fit_SH = randomForest::randomForest(formula = UsageRate_Gals_per_Hr ~ Days + Hours + Month  + Shift +
                                                 LastWeekSum + Last2WeekSum + Last3WeekSum  + Avg_Shift_Usage +
                                                 Avg_Day_Usage  + Avg_WeeklyShift_Usage + Avg_WeeklyDay_Usage +
                                                 Avg_Hourly_Usage_perMonth + Avg_Hourly_Usage + usage6hrs +
                                                 usage24hrs + usage60hrs,
                                               data = forecast_data,
                                               Importance = TRUE,
                                               ntree = 400)
        
        filename = paste(pathname, method, "_SH_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit_SH, file = filename)
     # } else {
        forecast_data = prepare_training_data(LocNum, Start_date, End_date, Region, Type, channelname)
        model_fit = randomForest::randomForest(y = forecast_data[, 1], x = forecast_data[, c(2: dim(forecast_data)[2])],
                                               Importance = TRUE,
                                               ntree = 400)
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        saveRDS(model_fit, file = filename)
     # }
        
      } else print("provide correct method name ")
    }
  }
  
  # filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
  # saveRDS(model_fit, file = filename)
  # return(RF_model)
}


get_LocNum = function(Region, Type, PrimaryTerminal, channelname){
  
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  if (PrimaryTerminal == "ALL") {
    LN_query = paste("SELECT Distinct [LocNum]
                     ,[PrimaryTerminal]
                     ,CP.[ProductClass]
                     ,TerminalTypes.TerminalType
                     ,Terminal.Region
                     ,PC.Gaseous
                     FROM ", DB," .[dbo].[CustomerProfile] CP
                     INNER JOIN ", DB," .[dbo].[TerminalTypes] ON PrimaryTerminal = TerminalTypes.CorporateIdn
                     INNER JOIN ", DB," .[dbo].[Terminal] ON PrimaryTerminal = Terminal.CorporateIdn
                     INNER JOIN ", DB," .[dbo].[ProductClass] PC ON CP.ProductClass = PC.ProductClass
                     WHERE TerminalType IN ( ", TerminalType, ")
                     AND Region IN (", Regionname, ") 
                     AND Gaseous = 0
                     ORDER BY LocNum")
  } else {
    LN_query = paste("SELECT Distinct [LocNum]
                     ,[PrimaryTerminal]
                     ,CP.[ProductClass]
                     ,TerminalTypes.TerminalType
                     ,Terminal.Region
                     ,PC.Gaseous
                     FROM ", DB," .[dbo].[CustomerProfile] CP
                     INNER JOIN ", DB," .[dbo].[TerminalTypes] ON PrimaryTerminal = TerminalTypes.CorporateIdn
                     INNER JOIN ", DB," .[dbo].[Terminal] ON PrimaryTerminal = Terminal.CorporateIdn
                     INNER JOIN ", DB," .[dbo].[ProductClass] PC ON CP.ProductClass = PC.ProductClass
                     WHERE TerminalType IN ( ", TerminalType, ")
                     AND Region IN (", Regionname, ") AND PrimaryTerminal IN ( ", PrimaryTerminal, ")
                     AND Gaseous = 0
                     ORDER BY LocNum")
  }
  #AND  PrimaryTerminal = '857'"
  #'", PrimaryTerminal,"'"
  channel = RODBC::odbcConnect(channelname)
  LN_Data = RODBC::sqlQuery(channel, LN_query)
  RODBC::odbcClose(channel)
  
  return(LN_Data)
  
}


prepare_test_data_shorthorizon = function(LocNum, Start_date, End_date, Region, Type, channelname){
  
  
  test_start_date = lubridate::ceiling_date(as.POSIXct(Start_date), unit = 'hour')
  test_end_date   = lubridate::ceiling_date(as.POSIXct(End_date), unit = 'hour')
  
  End_date        = Start_date
  Start_date      = as.POSIXct(Start_date, origin = "1970-01-01") - 24*60*60*60 # getting past 60 days data since start date
  #End_date        = as.POSIXct(End_date, origin = "1970-01-01") + 24*60*60*10   # forward 10 days data after end date
  
  
  Data1 = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname)
  
  if(Data1 != 101) {
    
    
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 6
    ind_S2 = Data1$Hours >= 7 & Data1$Hours <= 12
    ind_S3 = Data1$Hours >= 13 & Data1$Hours <= 18
    ind_S4 = Data1$Hours >= 19 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    Data1$Shift[ind_S4] = "Shift4"
    
    # Adding past 3 week usage information 
    #Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data1$weeklabel = ceiling(sort(1:dim(Data1)[1], decreasing = T)/168)
    
    # Data1 = Data1 %>% group_by(weeklabel) %>%
    #   mutate(WeekSum = sum(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_avgWeek = aggregate(UsageRate_Gals_per_Hr ~ weeklabel,
                             data = Data1, sum, na.rm = T)
    colnames(Data_avgWeek)[2] = "WeekSum"
    # ind_avgweek = match(Data1$Week_Label, Data_avgWeek$Week_Label)
    # Data1$WeekSum = Data_avgWeek$WeekSum[ind_avgweek]
    
    # cl = length(Data1$UsageRate_Gals_per_Hr)
    # LastWeekEnd = cl-168
    # Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    # 
    # Week2St = 2*168+1
    # Week2End = cl-2*168
    # Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    # 
    # Week3St = 3*168+1
    # Week3End = cl-3*168
    # Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    # 
    # Average Shift usage rate
    # Data1 = Data1 %>% group_by(Shift) %>%
    #   mutate(Avg_Shift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data1 = Data1 %>% group_by(Shift, weeklabel) %>%
    #   mutate(Avg_WeeklyShift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_weeklyshiftusage = aggregate(UsageRate_Gals_per_Hr ~ Shift + weeklabel, 
                                      data = Data1, mean, na.rm = T)
    colnames(Data_weeklyshiftusage)[3] = "Avg_WeeklyShift_Usage"
    
    Data_shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, 
                           data = Data1, mean, na.rm = T)
    colnames(Data_shift)[2] = "Avg_Shift_Usage"
    
    
    # Average Daily usage rate
    # Data1 = Data1 %>% group_by(Days, weeklabel) %>%
    #   mutate(Avg_WeeklyDay_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_weeklydayusage = aggregate(UsageRate_Gals_per_Hr ~ Days + weeklabel, 
                                    data = Data1, mean, na.rm = T)
    colnames(Data_weeklydayusage)[3] = "Avg_WeeklyDay_Usage"
    
    # Data1 = Data1 %>% group_by(Days) %>%
    #   mutate(Avg_Day_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_days = aggregate(UsageRate_Gals_per_Hr ~ Days, 
                          data = Data1, mean, na.rm = T)
    colnames(Data_days)[2] = "Avg_Day_Usage"
    
    # Data1 = Data1 %>% group_by(Days, Hours, Month) %>%
    #   mutate(Avg_Hourly_Usage_perMonth = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_hrpermonth = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours + Month, 
                                data = Data1, mean, na.rm = T)
    Data_hrpermonth = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours, 
                                data = Data_hrpermonth, mean, na.rm = T)
    
    colnames(Data_hrpermonth)[3] = "Avg_Hourly_Usage_perMonth"
    
    # Data1 = Data1 %>% group_by(Days, Hours) %>%
    #  mutate(Avg_Hourly_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_hrlyusage = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours, 
                               data = Data1, mean, na.rm = T)
    colnames(Data_hrlyusage)[3] = "Avg_Hourly_Usage"
    
    
    # 
    # Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    # colnames(Data_Day)[2] = "Avg_Day_Usage"
    # ind_Day = match(Data1$Days, Data_Day$Days)
    # Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    # 
    # # Average Monthly usage rate
    # Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    # colnames(Data_Month)[2] = "Avg_Month_Usage"
    # ind_month = match(Data1$Month, Data_Month$Month)
    # Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    # 
    # ---------------------------------------------------------------------
    # Creating the test data
    # --------------------------------------------------------------------
    
    time_seq  = seq(test_start_date, test_end_date, by = "hour")
    test_data = as.data.frame(time_seq)
    names(test_data) = "Time"
    
    test_data$Days  = weekdays(test_data$Time)
    test_data$Hours = lubridate::hour(test_data$Time)
    test_data$Month = lubridate::month(test_data$Time)
    
    ind_S1 = test_data$Hours >= 0 & test_data$Hours <= 6
    ind_S2 = test_data$Hours >= 7 & test_data$Hours <= 12
    ind_S3 = test_data$Hours >= 13 & test_data$Hours <= 18
    ind_S4 = test_data$Hours >= 19 & test_data$Hours <= 23
    
    test_data$Shift[ind_S1] = "Shift1"
    test_data$Shift[ind_S2] = "Shift2"
    test_data$Shift[ind_S3] = "Shift3"
    test_data$Shift[ind_S4] = "Shift4"
    
    test_data$weeklabel = 1
    
    test_data = inner_join(test_data, Data_shift, by = "Shift")
    test_data = inner_join(test_data, Data_hrlyusage, by = c("Hours", "Days"))
    test_data = inner_join(test_data, Data_days, by = "Days")
    test_data = inner_join(test_data, Data_hrpermonth, by = c("Hours", "Days"))
    test_data = inner_join(test_data, Data_weeklydayusage, by = c("Days", "weeklabel"))
    test_data = inner_join(test_data, Data_weeklyshiftusage, by = c("Shift", "weeklabel"))
    
    test_data$LastWeekSum  = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 1]
    test_data$Last2WeekSum = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 2]
    test_data$Last3WeekSum = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 3]
    
   usage6hrs = tail(Data1$UsageRate_Gals_per_Hr, 6)
    for (p in 1:length(time_seq)){
      usage6hrs[6+p] = mean(tail(usage6hrs, 6))
    }
   test_data$usage6hrs = tail(usage6hrs, -6)
     
   usage24hrs = tail(Data1$UsageRate_Gals_per_Hr, 24)
   for (p in 1:length(time_seq)){
     usage24hrs[24+p] = mean(tail(usage24hrs, 24))
   }
   test_data$usage24hrs = tail(usage24hrs, -24)
   
   usage60hrs = tail(Data1$UsageRate_Gals_per_Hr, 60)
   for (p in 1:length(time_seq)){
     usage60hrs[60+p] = mean(tail(usage60hrs, 60))
   }
   test_data$usage60hrs = tail(usage60hrs, -60)
   
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    #if (training == 0){
    #ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    # Data_F = Data1[ind_test, ]
    
    Data_F = test_data %>% select(Time, Days, Shift, Hours, Month, 
                                  LastWeekSum,Last2WeekSum, Last3WeekSum,
                                  Avg_Shift_Usage, Avg_WeeklyShift_Usage, Avg_WeeklyDay_Usage,
                                  Avg_Day_Usage, Avg_Hourly_Usage_perMonth, Avg_Hourly_Usage,
                                  usage6hrs, usage24hrs, usage60hrs)
    # usage6hrs, usage24hrs, usage60hrs)
    
    # } else  {
    #   ind_na = is.na(Data1$Last3WeekSum)
    #   Data_F = Data1[!ind_na, ]
    # }
    
    # Data_F = dplyr::select(Data_F, Time, UsageRate_Gals_per_Hr, Days, Shift, Hours,
    #                        Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
    #                        Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3, 4)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    
  } else return(101)
  
  # # --------------------------------------------------------------------------
  # # --------------------------------------------------------------------------
  # # Hourly Prediction 
  # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
  # Xvar_Data = Data_F
  # 
  # if (methodclass == 1){ return(Timeseries_Data)}
  # else if (methodclass == 2) { return(Xvar_Data)}
  # else print("Provide a method class")
}


prepare_test_data_longhorizon = function(LocNum, Start_date, End_date, Region, Type, channelname){
  
  
  test_start_date = lubridate::ceiling_date(as.POSIXct(Start_date), unit = 'hour')
  test_end_date   = lubridate::ceiling_date(as.POSIXct(End_date), unit = 'hour')
  
  End_date        = Start_date
  Start_date      = as.POSIXct(Start_date, origin = "1970-01-01") - 24*60*60*45 # getting past 45 days data since start date
  #End_date        = as.POSIXct(End_date, origin = "1970-01-01") + 24*60*60*10   # forward 10 days data after end date
  
  
  Data1 = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname)
  
  if(Data1 != 101) {
    
   
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 6
    ind_S2 = Data1$Hours >= 7 & Data1$Hours <= 12
    ind_S3 = Data1$Hours >= 13 & Data1$Hours <= 18
    ind_S4 = Data1$Hours >= 19 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    Data1$Shift[ind_S4] = "Shift4"
    
    # Adding past 3 week usage information 
    #Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data1$weeklabel = ceiling(sort(1:dim(Data1)[1], decreasing = T)/168)
    
    # Data1 = Data1 %>% group_by(weeklabel) %>%
    #   mutate(WeekSum = sum(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_avgWeek = aggregate(UsageRate_Gals_per_Hr ~ weeklabel,
                             data = Data1, sum, na.rm = T)
    colnames(Data_avgWeek)[2] = "WeekSum"
    # ind_avgweek = match(Data1$Week_Label, Data_avgWeek$Week_Label)
    # Data1$WeekSum = Data_avgWeek$WeekSum[ind_avgweek]
    
    # cl = length(Data1$UsageRate_Gals_per_Hr)
    # LastWeekEnd = cl-168
    # Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    # 
    # Week2St = 2*168+1
    # Week2End = cl-2*168
    # Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    # 
    # Week3St = 3*168+1
    # Week3End = cl-3*168
    # Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    # 
    # Average Shift usage rate
    # Data1 = Data1 %>% group_by(Shift) %>%
    #   mutate(Avg_Shift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    # Data1 = Data1 %>% group_by(Shift, weeklabel) %>%
    #   mutate(Avg_WeeklyShift_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_weeklyshiftusage = aggregate(UsageRate_Gals_per_Hr ~ Shift + weeklabel, 
                                      data = Data1, mean, na.rm = T)
    colnames(Data_weeklyshiftusage)[3] = "Avg_WeeklyShift_Usage"
    
    Data_shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, 
                           data = Data1, mean, na.rm = T)
    colnames(Data_shift)[2] = "Avg_Shift_Usage"
   
    
    # Average Daily usage rate
    # Data1 = Data1 %>% group_by(Days, weeklabel) %>%
    #   mutate(Avg_WeeklyDay_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_weeklydayusage = aggregate(UsageRate_Gals_per_Hr ~ Days + weeklabel, 
                           data = Data1, mean, na.rm = T)
    colnames(Data_weeklydayusage)[3] = "Avg_WeeklyDay_Usage"
    
    # Data1 = Data1 %>% group_by(Days) %>%
    #   mutate(Avg_Day_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_days = aggregate(UsageRate_Gals_per_Hr ~ Days, 
                           data = Data1, mean, na.rm = T)
    colnames(Data_days)[2] = "Avg_Day_Usage"
    
    # Data1 = Data1 %>% group_by(Days, Hours, Month) %>%
    #   mutate(Avg_Hourly_Usage_perMonth = mean(UsageRate_Gals_per_Hr, na.rm = T))
    
    Data_hrpermonth = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours + Month, 
                           data = Data1, mean, na.rm = T)
    Data_hrpermonth = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours, 
                                data = Data_hrpermonth, mean, na.rm = T)
   
    colnames(Data_hrpermonth)[3] = "Avg_Hourly_Usage_perMonth"
    
     # Data1 = Data1 %>% group_by(Days, Hours) %>%
     #  mutate(Avg_Hourly_Usage = mean(UsageRate_Gals_per_Hr, na.rm = T))
     
     Data_hrlyusage = aggregate(UsageRate_Gals_per_Hr ~ Days + Hours, 
                            data = Data1, mean, na.rm = T)
     colnames(Data_hrlyusage)[3] = "Avg_Hourly_Usage"
    
    # Data1$usage6hrs = NA
    # for (p in 7:dim(Data1)[1]){
    #   T1 = Data1$Time[p-6]
    #   T2 = Data1$Time[p]
    #   Data1$usage6hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data1$usage24hrs = NA
    # for (p in 25:dim(Data1)[1]){
    #   T1 = Data1$Time[p-24]
    #   T2 = Data1$Time[p]
    #   Data1$usage24hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data1$usage60hrs = NA
    # for (p in 61:dim(Data1)[1]){
    #   T1 = Data1$Time[p-60]
    #   T2 = Data1$Time[p]
    #   Data1$usage60hrs[p] = mean(Data1$UsageRate_Gals_per_Hr[Data1$Time >= T1 & Data1$Time < T2 ])
    # }
    # 
    # Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    # colnames(Data_Day)[2] = "Avg_Day_Usage"
    # ind_Day = match(Data1$Days, Data_Day$Days)
    # Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    # 
    # # Average Monthly usage rate
    # Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    # colnames(Data_Month)[2] = "Avg_Month_Usage"
    # ind_month = match(Data1$Month, Data_Month$Month)
    # Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    # 
    # ---------------------------------------------------------------------
    # Creating the test data
    # --------------------------------------------------------------------
    
    time_seq  = seq(test_start_date, test_end_date, by = "hour")
    test_data = as.data.frame(time_seq)
    names(test_data) = "Time"
    
    test_data$Days  = weekdays(test_data$Time)
    test_data$Hours = lubridate::hour(test_data$Time)
    test_data$Month = lubridate::month(test_data$Time)
    
    ind_S1 = test_data$Hours >= 0 & test_data$Hours <= 6
    ind_S2 = test_data$Hours >= 7 & test_data$Hours <= 12
    ind_S3 = test_data$Hours >= 13 & test_data$Hours <= 18
    ind_S4 = test_data$Hours >= 19 & test_data$Hours <= 23
    
    test_data$Shift[ind_S1] = "Shift1"
    test_data$Shift[ind_S2] = "Shift2"
    test_data$Shift[ind_S3] = "Shift3"
    test_data$Shift[ind_S4] = "Shift4"
    
    test_data$weeklabel = 1
    
    test_data = inner_join(test_data, Data_shift, by = "Shift")
    test_data = inner_join(test_data, Data_hrlyusage, by = c("Hours", "Days"))
    test_data = inner_join(test_data, Data_days, by = "Days")
    test_data = inner_join(test_data, Data_hrpermonth, by = c("Hours", "Days"))
    test_data = inner_join(test_data, Data_weeklydayusage, by = c("Days", "weeklabel"))
    test_data = inner_join(test_data, Data_weeklyshiftusage, by = c("Shift", "weeklabel"))
    
    test_data$LastWeekSum  = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 1]
    test_data$Last2WeekSum = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 2]
    test_data$Last3WeekSum = Data_avgWeek$WeekSum[Data_avgWeek$weeklabel == 3]
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    #if (training == 0){
    #ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    # Data_F = Data1[ind_test, ]
    
    Data_F = test_data %>% select(Time, Days, Shift, Hours, Month, 
                               LastWeekSum,Last2WeekSum, Last3WeekSum,
                               Avg_Shift_Usage, Avg_WeeklyShift_Usage, Avg_WeeklyDay_Usage,
                               Avg_Day_Usage, Avg_Hourly_Usage_perMonth, Avg_Hourly_Usage)
                              # usage6hrs, usage24hrs, usage60hrs)
    
    # } else  {
    #   ind_na = is.na(Data1$Last3WeekSum)
    #   Data_F = Data1[!ind_na, ]
    # }
    
    # Data_F = dplyr::select(Data_F, Time, UsageRate_Gals_per_Hr, Days, Shift, Hours,
    #                        Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
    #                        Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3, 4)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    
  } else return(101)
  
  # # --------------------------------------------------------------------------
  # # --------------------------------------------------------------------------
  # # Hourly Prediction 
  # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
  # Xvar_Data = Data_F
  # 
  # if (methodclass == 1){ return(Timeseries_Data)}
  # else if (methodclass == 2) { return(Xvar_Data)}
  # else print("Provide a method class")
}


get_usage_rate_test_data = function(LocNum, Start_date, End_date, Region, Type, channelname)  {
  
  # packages 
  require(RODBC)
  require(dplyr)
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  # --------------------------------------------------------------------
  
  
  # Getting the customer profile data
  CP_sql  = CP_sql = paste("SELECT [LocNum]
                           ,[ProductAbbrev]
                           ,[GalsPerInch]
                           FROM ", DB,".[dbo].[CustomerProfile]
                           WHERE Locnum = '", LocNum, "'")
  #CP_sql1 = paste(paste(CP_sql[CP_sql!='--'],collapse = ' '), "WHERE Locnum = '", LocNum, "'")
  
  
  # Telemetry Reading Data
  # Reading data from now to 6 months back
  # TR_end_time   = Sys.time() # Current time
  # TR_start_time = seq(as.Date(TR_end_time), length = 2, by = "-6 months")[2] # 6 months back from today
  # TR_start_time = as.POSIXct(TR_start_time)
  TR_start_time = as.POSIXct(Start_date, origin = "1970-01-01")
  TR_end_time   = as.POSIXct(End_date, origin = "1970-01-01")
  
  TR_sql = paste("SELECT R.[AccountNum]
                 ,R.[LocNum]
                 ,[ReadingDate]
                 ,CASE WHEN [ReadingUOM] = 1 THEN [ReadingLevel] / [GalsPerInch] ELSE [ReadingLevel] END AS ReadingLevel
                 ,[ReadingType]
                 FROM", DB,".[dbo].[Readings] R
                 JOIN", DB,".[dbo].[CustomerProfile] CP
                 ON R.LocNum = CP.LocNum
                 WHERE ReadingDate >= '", TR_start_time, "'",
                 "AND ReadingDate <='", TR_end_time, "'",
                 "AND (ReadingLevel > 0 or (readinglevel = 0 and readingtype = 2))
                 AND ReadingUOM != 2
                 AND R.Locnum = '", LocNum, "'",
                 "AND ReadingStatus != 2
                 ORDER BY ReadingDate")
  
  channel = RODBC::odbcConnect(channelname)
  CP_Data = RODBC::sqlQuery(channel, CP_sql)
  TR_Data = RODBC::sqlQuery(channel, TR_sql)
  RODBC::odbcClose(channel)
  # 
  
  # TR_Data = get_actual_readings(LocNum, Start_date, End_date, Region, Type, channelname)
  # ------------------------------------------------------------------------- 
  # Running only if at least 3 telemetry data exist for the customer
  if(dim(TR_Data)[1] > 3) {
    # Adding reading in gals
    TR_Data$Reading_Gals = TR_Data$ReadingLevel*CP_Data$GalsPerInch
    
    # Getting the customer profile data
    CusProfile_Reading = get_customer_profile(LocNum, Region, Type, channelname)
    
    
    # Creating the time sequence for the length of dataset (exclude timeset until first data is found)
    # last_data_time = tail(TR_Data$ReadingDate, 1)
    # first_data_time = round(head(TR_Data$ReadingDate, 1), units = "hours") - 60*60 # first hour closest to the first reading
    # data_length_hours = round(as.numeric(difftime(last_data_time, first_data_time, units = "hours")))
    # Time_Seq = seq(first_data_time, by = "hour", length.out = data_length_hours)
    data_length_hours = round(as.numeric(difftime(TR_end_time, TR_start_time, units = "hours")))
    Time_Seq = seq(TR_start_time, by = "hour", length.out = data_length_hours)
    
    #Reading = head(rep(Reading, 105), 17545)
    Time_Seq = as.data.frame(Time_Seq)
    names(Time_Seq) = "Time"
    
    CusProfile = Time_Seq %>%
      mutate(weekdays = weekdays(Time),
             hours    = lubridate::hour(Time),
             LocNum   = LocNum) %>%
      left_join(CusProfile_Reading, by = c("weekdays", "hours"))
    
    
    # ------------------------------------------------------------------------------
    # Telemetry Reading Data
    # ------------------------------------------------------------------------------
    
    TR_Data = TR_Data[order(TR_Data$ReadingDate), ]
    LinearApprox = approx(TR_Data$ReadingDate,
                          TR_Data$Reading_Gals,
                          Time_Seq$Time, method = "linear", rule = 2)
    
    CusProfile$LinearReadings = LinearApprox$y
    CusProfile$LinearUsage = round(-c(0,diff(CusProfile$LinearReadings)),3)
    CusProfile$LinearUsage[CusProfile$LinearUsage < 0] = NA
    
    CusProfile$LinearUsage[is.na(CusProfile$LinearUsage) == 1] = median(CusProfile$LinearUsage, na.rm = T)
    
    # Removing outliers | replacing the value with mean (double pass)
    # n.pass = 0
    # for (i in 1:n.pass) {
    #   ind_nonzero = CusProfile$LinearUsage > 0
    #   HighVal = mean(CusProfile$LinearUsage[ind_nonzero]) + 3*sd(CusProfile$LinearUsage[ind_nonzero])
    #   CusProfile$LinearUsage[CusProfile$LinearUsage > HighVal] = median(CusProfile$LinearUsage, na.rm = T)
    # }
    
    # Converting hourly readings to UsageRate (Gals/Hr) 
    # Using the customer profile data between each reading
    
    Reading_Time = unique(as.POSIXct(TR_Data$ReadingDate), na.rm = T)
    Reading_Time = Reading_Time[is.na(Reading_Time) == 0]
    for (i in 2:length(Reading_Time)){
      ind_BR = (CusProfile$Time > Reading_Time[i-1] & CusProfile$Time <= Reading_Time[i])
      Num_Readings = sum(CusProfile$Reading[ind_BR])
      if ((Num_Readings/100 == sum(ind_BR)) | (Num_Readings == 0)) {
        CusProfile$UsageRate_Gals_per_Hr[ind_BR] = CusProfile$LinearUsage[ind_BR]
      } else {
        CusProfile$UsageRate_Gals_per_Hr[ind_BR] = (sum(CusProfile$LinearUsage[ind_BR]))*
          (CusProfile$Reading[ind_BR]/Num_Readings)
      }
    }
    ind_na = is.na(CusProfile$UsageRate_Gals_per_Hr)
    CusProfile$UsageRate_Gals_per_Hr[ind_na] = CusProfile$LinearUsage[ind_na]
    CusProfile$UsageRate_Gals_per_Hr = round(CusProfile$UsageRate_Gals_per_Hr, 3)
    
    CusProfile = CusProfile %>%
      select(LocNum, Time, Reading, LinearReadings, LinearUsage, UsageRate_Gals_per_Hr)
    
    names(CusProfile) = c("LocNum", "Time", "CustomerPattern", "LinearUsage_Gals",
                          "LinearUsage_Gals_per_Hr", "UsageRate_Gals_per_Hr")
    return(CusProfile)
    
  } else {
    # print(paste("No Telemetry data for LocNum = ", LocNum))
    # break
    
    CusProfile = 101
  }
} 


get_actual_readings = function(LocNum, Start_date, End_date, Region, Type, channelname)  {
  
  
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  # --------------------------------------------------------------------
  
  # CP_sql  = CP_sql = paste("SELECT [LocNum]
  #                          ,[ProductAbbrev]
  #                          ,[GalsPerInch]
  #                          ,[RunoutGals]
  #                          FROM ", DB,".[dbo].[CustomerProfile] 
  #                          WHERE Locnum = '", LocNum, "'")


  TR_start_time = as.POSIXct(Start_date, origin = "1970-01-01")
  TR_end_time   = as.POSIXct(End_date, origin = "1970-01-01")
  
  TR_sql = paste("SELECT R.[LocNum]
                 ,[ReadingDate]
                 ,CP.[RunoutGals]
                 ,CASE WHEN [ReadingUOM] = 1 THEN [ReadingLevel]/[GalsPerInch] ELSE [ReadingLevel] END AS ReadingLevel
                 ,[ReadingType]
                 ,CASE WHEN [ReadingUOM] = 0 THEN [ReadingLevel] * [GalsPerInch] ELSE [ReadingLevel] END AS Reading_Gals
                 ,", DB,".dbo.GetActualDlvyAmount(CorporateIdn, TripIdn, SegmentIdn) AS Actual_Delivered_Amount
                 FROM ", DB,".[dbo].[Readings] R
                 JOIN ", DB,".[dbo].[CustomerProfile] CP 
                 ON R.LocNum = CP.LocNum
                 LEFT JOIN ", DB," .[dbo].[Segment] Seg
                 on Seg.ToLocNum = R.LocNum and Seg.ActualArrivalTime = R.ReadingDate
                 WHERE ReadingDate >= '", TR_start_time, "'",
                 "AND ReadingDate <'", TR_end_time, "'",
                 "AND (ReadingLevel > 0 or (readinglevel = 0 and readingtype = 2))
                 AND ReadingUOM != 2
                 AND R.Locnum = '", LocNum, "'",
                 "AND ReadingStatus != 2
                 ORDER BY ReadingDate")
  
  # Segment_sql = paste("SELECT [ToLocNum] As LocNum
  #                     ,[ActualArrivalTime] As ReadingDate
  #                     ,", DB,".dbo.GetActualDlvyAmount(CorporateIdn, TripIdn, SegmentIdn) AS Actual_Delivered_Amount
  #                     FROM ", DB," .[dbo].[Segment]
  #                     WHERE ToLocNum = '", LocNum, "' AND ActualArrivalTime >= '", TR_start_time,"' 
  #                     AND ActualArrivalTime <= '", TR_end_time,"'
  #                     ORDER BY ActualArrivalTime")
  # 

  channel = RODBC::odbcConnect(channelname)
  #CP_Data = RODBC::sqlQuery(channel, CP_sql)
  TR_Data = RODBC::sqlQuery(channel, TR_sql, as.is = T)
  #Segment_Data = RODBC::sqlQuery(channel, Segment_sql)
  
  RODBC::odbcClose(channel)
  
  if(dim(TR_Data)[1] > 0) {
    TR_Data$ReadingDate  = as.POSIXct(TR_Data$ReadingDate, tz = 'EST')
    TR_Data$ReadingLevel = as.numeric(TR_Data$ReadingLevel)
    TR_Data$Reading_Gals = as.numeric(TR_Data$Reading_Gals)
    # TR_Data$Reading_Gals = TR_Data$ReadingLevel*CP_Data$GalsPerInch
    # if (dim(Segment_Data)[1] >0){
    # TR_Data = TR_Data %>%
    #   left_join(Segment_Data, by = c("LocNum", "ReadingDate"))
    # } else 
    # TR_Data$Actual_Delivered_Amount = 0  
  } 
  # else {
  #   TR_Data$Actual_Delivered_Amount = NA # Proxy for no reading info 
  # }
  # 
  return(TR_Data)
}


get_forecast_level = function(LocNum, Start_date, End_date, Region, Type, channelname, pathname, Method){
  
  #cus_stat_data = get_reading_statistics(LocNum, Start_date, channelname)
  
  
  
  Actual_data = get_actual_readings(LocNum, Start_date, End_date, Region, Type, channelname)
  #Actual_data = Actual_data %>% filter(ReadingType == 2 | ReadingType == 3) # Only for delivery forecast
  
  if (dim(Actual_data)[1] > 0 & dim(Actual_data)[2] > 4){ 
    # Checking if actual data is present or not and there is a level information 
    # And if segment data available for the forecast dates 
    
    # Getting the forecast usage rate 
    Forecast_data    = get_forecast(LocNum, Start_date, End_date, Region, Type, channelname, pathname, Method)
    LBShell_forecast = get_LBShell_forecast(LocNum, Start_date, End_date, Region, Type, channelname)
    
    
    if (any(is.na(Forecast_data)) == 0) { # Checking if the forecast is present or not 
      
      #if (dim(Actual_data)[2] > 4) { # No segment data available for the forecast dates 
        
          Actual_data$hourly_time = as.POSIXct(round(Actual_data$ReadingDate, units = "hours"))
          # Time differenc between readings
          Actual_data$Days_btw_readings = round(c(0, diff(Actual_data$ReadingDate)/(24*60)), 2)
          # Time difference between rounded time and reading time
          Actual_data$Time_diff = difftime(Actual_data$hourly_time, Actual_data$ReadingDate, units = "mins")
          
          if (LBShell_forecast != 101){
            Method = c(Method, "LBShell")
            Forecast_data = inner_join(Forecast_data, LBShell_forecast, by = "Time")
          }
          
          
          for (m_ind in 1:length(Method)) {
            Actual_data$forecasted_level = 0
            
            forecast_usage_rate = Forecast_data[, m_ind + 2] # NEED TO BE CAREFUL AS STRUCTURE CAN CHANGE
            # Computing the forecasted level
            for (i in 1: (dim(Actual_data)[1]-1)){
              T1 = Actual_data$hourly_time[i]
              T2 = Actual_data$hourly_time[i+1]
              ind_time = Forecast_data$Time >= T1 & Forecast_data$Time <= T2
              usage = sum(forecast_usage_rate[ind_time]) + 
                as.numeric(Actual_data$Time_diff[i]/60)*forecast_usage_rate[Forecast_data$Time == T1] -
                as.numeric(Actual_data$Time_diff[i+1]/60)*forecast_usage_rate[Forecast_data$Time == T2]
              Days_btw_readings = Actual_data$ReadingDate[i] 
              Actual_data$forecasted_level[i+1] = Actual_data$Reading_Gals[i] - usage
            }
            
            Actual_data$forecasted_level[Actual_data$forecasted_level < 0] = 0
            method = Method[m_ind]
            names(Actual_data)[names(Actual_data) == 'forecasted_level'] <- paste(method, "_forecasted_level", sep = "")
          }
          if (LBShell_forecast == 101) {Actual_data$LBShell_forecasted_level = NA}
          # Removing the first record and record with reading type = 3
          Actual_data = Actual_data[2:dim(Actual_data)[1], ]
          Actual_data = Actual_data %>%
            filter(ReadingType != 3)
          
          # Absolute percentage error 
          # Actual_data$forecast_level_APE = (abs(Actual_data$Reading_Gals - Actual_data$forecasted_level)/Actual_data$Reading_Gals)*100
          Actual_data = Actual_data[, -c(7,8,9)] # Removing unwanted column
          
      # } else { # No segment data available for the forecast dates 
      #   Actual_data$Reading_Gals = NA
      #   Actual_data$Actual_Delivered_Amount = NA
      #   Actual_data$forecasted_level = NA
      #   Actual_data$forecast_level_APE = NA
      # }
      Output = Actual_data
      #return(Output)
    } else {
      
      # If no forecast data exist
      Actual_data$NN_forecasted_level = NA
      Actual_data$ARIMA_forecasted_level = NA
      Actual_data$ETS_forecasted_level = NA
      Actual_data$RF_forecasted_level = NA
      Actual_data$ARIMAX_forecasted_level = NA
      Actual_data$NNX_forecasted_level = NA
      Actual_data$LBShell_forecasted_level = NA
      #Actual_data$forecast_level_APE = NA
      Output = Actual_data
      #return(Output)
    } 
     
  } else {
    
    # If no actual data exist 
    Output = data.frame(LocNum)
    Output$ReadingDate  = NA
    Output$ReadingLevel = NA
    Output$ReadingType = NA
    Output$Reading_Gals = NA
    Output$Actual_Delivered_Amount = NA
    Output$RF_forecasted_level = NA
    Output$ARIMAX_forecasted_level = NA
    Output$NNX_forecasted_level = NA
    Output$NN_forecasted_level = NA
    Output$ARIMA_forecasted_level = NA
    Output$ETS_forecasted_level = NA
    Output$LBShell_forecasted_level = NA
    #Output$forecast_level_APE = NA
  }
  
  
  return(Output)
}


get_forecast = function(LocNum, Start_date, End_date, Region, Type, channelname, pathname, Method) {

    forecast_data = prepare_test_data(LocNum, Start_date, End_date, Region, Type, channelname)
    # Need to include a line to choose a single method and probably loop for the numbers of method
    for (i in 1:length(Method)) {
      method = Method[i]
      
      if(forecast_data == 101) {
        if (i == 1) {
          # No forecast dats
          Output = data.frame(LocNum)
          Output$Time = Start_date
        }
        #Output$Method = method
        Output$forecast_usage_rate = NA
        names(Output)[names(Output) == 'forecast_usage_rate'] <- paste(method, "_forecast_usage_rate", sep = "")
      } else {
      
      #pathname = "\\\\home7\\DecisionScience\\Anshu Gupta\\LBShell Forecast\\model\\"
      filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
      if (file.exists(filename) == 1){
        # Using existing model
        model_fit = readRDS(filename)
        if (typeof(model_fit) == "character") {

          print(model_fit) # No past data avaulable for model
          if (i == 1) {
            Output = data.frame(forecast_data$Time)
            Output$LocNum = LocNum
          }
          #Output$Method = method
          Output$forecast_usage_rate = NA
          names(Output)[names(Output) == 'forecast_usage_rate'] <- paste(method, "_forecast_usage_rate", sep = "")
          #Output$Comment = model_fit

        } else {


          # If data is available to forecast (101 is used as a flag to show no data)
          if (forecast_data != 101) {


            if (method == "ARIMA")  {
              usage_data = ts(forecast_data$UsageRate_Gals_per_Hr)
              pred_horizon = 24 # 6 hours - can chnage this
              v1 = 1
              v2 = v1 + pred_horizon
              model_forecast = foreach(p = seq(from = 1, to = dim(forecast_data)[1]-2, by = pred_horizon),
                                       .combine = 'rbind', .packages = c("forecast")) %do% {

                                         model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
                                         model_fit = Arima(usage_data[v1:v2], model = model_fit)

                                         v1 = v1 + pred_horizon
                                         v2 = ifelse(v2 + pred_horizon > dim(forecast_data)[1], dim(forecast_data)[1], v2 + pred_horizon)
                                         # print(v1)
                                         # print(v2)
                                         print(model_forecast)
                                         #model_forecast = model_forecast_1
                                       }
              model_forecast = model_forecast$`Point Forecast`
              if (dim(forecast_data)[1] > length(model_forecast)){
                model_forecast = c(model_forecast, tail(model_forecast, dim(forecast_data)[1] - length(model_forecast)))
              }
              #ifelse(length(model_forecast) < dim(forecast_data[1], ))
            } else if ( method == "ETS" ) {

              usage_data = ts(forecast_data$UsageRate_Gals_per_Hr)
              pred_horizon = 24 # 6 hours - can chnage this
              v1 = 1
              v2 = v1 + pred_horizon
              model_forecast = foreach(p = seq(from = 1, to = dim(forecast_data)[1]-2, by = pred_horizon),
                                       .combine = 'rbind', .packages = c("forecast")) %do% {

                                         model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
                                         model_fit = ets(usage_data[v1:v2], model = model_fit, use.initial.values = T)

                                         v1 = v1 + pred_horizon
                                         v2 = ifelse(v2 + pred_horizon > dim(forecast_data)[1], dim(forecast_data)[1], v2 + pred_horizon)
                                         # print(v1)
                                         # print(v2)
                                         print(model_forecast)
                                       }
              model_forecast = model_forecast$`Point Forecast`
              if (dim(forecast_data)[1] > length(model_forecast)){
                model_forecast = c(model_forecast, tail(model_forecast, dim(forecast_data)[1] - length(model_forecast)))
              }

            } else if ( method == "NN") {
              pred_horizon = dim(forecast_data)[1]
              model_forecast = forecast::forecast(model_fit, h = pred_horizon)
              model_forecast = model_forecast$mean

            } else if (method == "ARIMAX" | (method == "NNX"))  {
              pred_horizon = dim(forecast_data)[1]
              xVars           = data.matrix(forecast_data[, c(3: dim(forecast_data)[2])])
              model_forecast  = forecast::forecast(model_fit, xreg = xVars, h = pred_horizon)
              model_forecast  = model_forecast$mean
              # } else if (method == "ETS") {
              #   # Exponential smooting
              #   model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
              # } else if (method == "NN") {
              #   # Neural network
              #   model_forecast      = forecast::forecast(model_fit, h = pred_horizon)
              # } else if (method == "NNX") {
              #   # Neural network with x variable
              #   xVars           = data.matrix(forecast_data[, c(2: dim(forecast_data)[2])])
              #   model_fit       = forecast::nnetar(ts(forecast_data[, 1]), xreg = xVars, MaxNWts = 3000)
            } else if (method == "RF") {
              # levels(forecast_data$Days) = 1:7
              # levels(forecast_data$Shift) = 1:3
              # levels(forecast_data$Hours) = 0:23
              # levels(forecast_data$Month) = 1:12
              model_forecast = predict(model_fit, forecast_data[, c(3:dim(forecast_data)[2])] )
            }

            if (i == 1) {
              Output = data.frame(forecast_data$Time)
              names(Output) = "Time"
              Output$LocNum = LocNum
            }
            # Adding the forecast and changing the name to match the current forecast method
            Output$forecast_usage_rate = model_forecast


            #forecast_data$LocNum = LocNum
            #forecast_data$Comment = ""
            # Output = forecast_data %>%
            #   select(Time, LocNum, Method, forecast_usage_rate)

            names(Output)[names(Output) == 'forecast_usage_rate'] <- paste(method, "_forecast_usage_rate", sep = "")

          } else {

            if (i == 1) {
              # No forecast dats
              Output = data.frame(LocNum)
              Output$Time = Start_date
            }
            #Output$Method = method
            Output$forecast_usage_rate = NA
            names(Output)[names(Output) == 'forecast_usage_rate'] <- paste(method, "_forecast_usage_rate", sep = "")
            #Output$Comment = paste("No current data for", LocNum, "between", Start_date, End_date)

          }

        }
        #return(101)
      } else {
        # Need to create a new model
        if (i == 1) {
          Output = data.frame(forecast_data$Time)
          Output$LocNum = LocNum
        }
        #Output$Method = method
        Output$forecast_usage_rate = NA
        names(Output)[names(Output) == 'forecast_usage_rate'] <- paste(method, "_forecast_usage_rate", sep = "")
        #Output$Comment = "No current model in directory - fit new model"
        #return(101)
        # End_date = Sys.Date()
        # Start_date = seq(as.Date(End_date), length = 2, by = "-12 months")[2] # 12 months back from today
        # model_fit = train_forecast_model(LocNum, Start_date, End_date, Region, Type, channelname, method)
      }
      }
    }

    return(Output)
}



get_prediction_horizon_forecast = function(LocNum,  Test_start_date, Test_end_date, Region, Type,
                                   channelname, pred_horizon, pathname){
  # Runs for all the method

  Delivery_Data = tryCatch(
    {
      
      # Getting the prediction over the prediction horizon
      T1 = as.POSIXct(Test_start_date) - 168*60*60 # go 1 week back
      T2 = as.POSIXct(Test_end_date)
      Actual_data = get_actual_readings(LocNum, T1, T2, Region, Type, channelname)
      Delivery_Data = Actual_data %>% filter(ReadingType == 2, ReadingDate >= Test_start_date)


      for (g in 1:dim(Delivery_Data)[1]){
    
     
        T2 = Delivery_Data$ReadingDate[g]
        A1 = Actual_data %>% filter(ReadingDate <= T2)
        
        Time_diff = as.numeric(difftime(A1$ReadingDate[-1], A1$ReadingDate[-nrow(A1)], units="hours"))
        
        for (h in 1:length(Time_diff)){
          total_hrs = sum(tail(Time_diff, h))
          if (total_hrs > pred_horizon) break
        }
        
        # Reading from where delivery reading is computed
        T1 = A1$ReadingDate[length(Time_diff) - h + 1]
        L1 = A1$Reading_Gals[A1$ReadingDate == T1]
        Delivery_Data$Time_Predicted_from[g] = T1
        Delivery_Data$Level_predicted_from[g] = L1
        
        # Finding any delivery between the period
        A2 = A1 %>% filter(ReadingDate > T1)
        A2 = head(A2, -1)
        A2 = A2 %>% filter (ReadingType == 2)
        Delivery_Data$amount_delivered[g] = sum(A2$Actual_Delivered_Amount)  
        
        # Defining the prediction horizon
        pred_hr = ceiling(as.numeric(difftime(T2, T1, units = "hours")))
        Delivery_Data$Time_diff[g] = pred_hr
        
        # LBShell
        LBShell_data = get_LBShell_forecast(LocNum, T1, T2, Region, Type, channelname)
        if (LBShell_data != 101){
        total_usage = sum(LBShell_data$LBShell, na.rm = T)
        Delivery_Data$LBShell_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        } else Delivery_Data$LBShell_predicted_level[g] = NA
        
        # ARIMA
        method = "ARIMA"
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$ARIMA_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        
        # ETS
        method = "ETS"
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$ETS_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        
        # NN
        method = "NN"
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$NN_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        
        # Forecast data
        forecast_data = prepare_test_data_shorthorizon(LocNum, T1, T2, Region, Type, channelname)
        xVars  = data.matrix(forecast_data[, c(2: dim(forecast_data)[2])])
        
        # NNX
        # method = 'NNX'
        # filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        # model_fit = readRDS(filename)
        # forecast_usage_rate = predict(model_fit, xreg = xVars )
        # forecast_usage_rate[forecast_usage_rate < 0] = 0
        # total_usage = sum(forecast_usage_rate, na.rm = T)
        # Delivery_Data$NNX_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        # 
        # # ARIMAX
        # method = 'ARIMAX'
        # filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        # model_fit = readRDS(filename)
        # forecast_usage_rate = predict(model_fit, xreg = xVars )
        # forecast_usage_rate[forecast_usage_rate < 0] = 0
        # total_usage = sum(forecast_usage_rate, na.rm = T)
        # Delivery_Data$ARIMAX_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        
        # Random Forest
        method = "RF"
        
        # Short horizon RF
         filename = paste(pathname, method, "_SH_model_", LocNum, ".rds", sep = "")
         model_fit = readRDS(filename)
         forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
         forecast_usage_rate[forecast_usage_rate < 0] = 0
         total_usage = sum(forecast_usage_rate, na.rm = T)
         Delivery_Data$RF_SH_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
 
         # Long horizon model 
         filename = paste(pathname, method, "_LH_model_", LocNum, ".rds", sep = "")
         model_fit = readRDS(filename)
         forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
         forecast_usage_rate[forecast_usage_rate < 0] = 0
         total_usage = sum(forecast_usage_rate, na.rm = T)
         Delivery_Data$RF_LH_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage

         # Original RF version 
         forecast_data = prepare_test_data(LocNum,T1, T2, Region, Type, channelname)
         filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
         model_fit = readRDS(filename)
         forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
         forecast_usage_rate[forecast_usage_rate < 0] = 0
         total_usage = sum(forecast_usage_rate, na.rm = T)
         Delivery_Data$RF_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
     
        }
    
      Delivery_Data$Time_Predicted_from = as.POSIXct(Delivery_Data$Time_Predicted_from, origin = '1970-01-01')
      Delivery_Data = Delivery_Data

    }, 
 error = function(cond) {
   Delivery_Data = as.data.frame(LocNum)
   Delivery_Data$ReadingDate = NA
   Delivery_Data$ReadingLevel = NA
   Delivery_Data$Reading_Gals = NA
   Delivery_Data$Time_Predicted_from = NA
   Delivery_Data$Level_predicted_from = NA
   Delivery_Data$amount_delivered = NA
   Delivery_Data$Time_diff = NA
   Delivery_Data$LBShell_predicted_level = NA
   Delivery_Data$ARIMA_predicted_level = NA
   Delivery_Data$ETS_predicted_level = NA
   Delivery_Data$NN_predicted_level = NA
   # Delivery_Data$NNX_predicted_level = NA
   # Delivery_Data$ARIMA_predicted_level = NA
   Delivery_Data$RF_SH_predicted_level = NA
   Delivery_Data$RF_LH_predicted_level = NA
   Delivery_Data$RF_predicted_level = NA
   return(Delivery_Data)
 }
   
  )

Output = Delivery_Data %>% select(LocNum, ReadingDate, Time_Predicted_from, Level_predicted_from, 
                                  amount_delivered, Time_diff,
                                  ReadingLevel, Reading_Gals, LBShell_predicted_level,
                                  ARIMA_predicted_level,
                                  ETS_predicted_level, NN_predicted_level, 
                                  RF_SH_predicted_level, RF_LH_predicted_level,
                                  RF_predicted_level)


}

get_prediction_horizon_forecast_V2 = function(LocNum,  Test_start_date, Test_end_date, Region, Type,
                                           channelname, pred_horizon, pathname, method, RF_version){
  # Only the provided method 
  Delivery_Data = tryCatch(
    {
      # Getting the prediction over the prediction horizon
      T1 = as.POSIXct(Test_start_date) - pred_horizon*60*60
      T2 = as.POSIXct(Test_end_date)
      Actual_data = get_actual_readings(LocNum, T1, T2, Region, Type, channelname)
      #Delivery_Data = Actual_data %>% filter(ReadingType == 2, ReadingDate >= Test_start_date)
      ind_del = (Actual_data$ReadingType == 2 & Actual_data$ReadingDate >= Test_start_date)
      Delivery_Data = Actual_data[which(ind_del) - 1, ]
      
      for (g in 1:dim(Delivery_Data)[1]){
        
        T2 = Delivery_Data$ReadingDate[g]
        A1 = Actual_data %>% filter(ReadingDate <= T2)
        
        Time_diff = as.numeric(difftime(A1$ReadingDate[-1], A1$ReadingDate[-nrow(A1)], units="hours"))
        
        for (h in 1:length(Time_diff)){
          total_hrs = sum(tail(Time_diff, h))
          if (total_hrs > pred_horizon) break
        }
        
        # Reading from where delivery reading is computed
        T1 = A1$ReadingDate[length(Time_diff) - h + 1]
        L1 = A1$Reading_Gals[A1$ReadingDate == T1]
        Delivery_Data$Time_Predicted_from[g] = T1
        Delivery_Data$Level_predicted_from[g] = L1
        
        # Finding any delivery between the period
        A2 = A1 %>% filter(ReadingDate > T1)
        A2 = head(A2, -1)
        A2 = A2 %>% filter (ReadingType == 2)
        Delivery_Data$amount_delivered[g] = sum(A2$Actual_Delivered_Amount)  
        
        # Defining the prediction horizon
        pred_hr = ceiling(as.numeric(difftime(T2, T1, units = "hours")))
        Delivery_Data$Time_diff[g] = pred_hr
        
        # LBShell
        LBShell_data = get_LBShell_forecast(LocNum, T1, T2, Region, Type, channelname)
        if (LBShell_data != 101){
          total_usage = sum(LBShell_data$LBShell, na.rm = T)
          #Delivery_Data$LBShell_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
          Delivery_Data$LBShell_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
        } else Delivery_Data$LBShell_predicted_level[g] = NA
        
        if (method == "ARIMA"){
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$ARIMA_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
        } else if (method == "ETS"){
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$ETS_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
        } else if (method == "NN") {
        filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
        model_fit = readRDS(filename)
        forecast_usage_rate = forecast::forecast(model_fit, h = pred_hr)
        forecast_usage_rate = forecast_usage_rate$mean
        forecast_usage_rate[forecast_usage_rate < 0] = 0
        total_usage = sum(forecast_usage_rate, na.rm = T)
        Delivery_Data$NN_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
        } else if (method == "RF"){
        if (RF_version == 2){
          forecast_data = prepare_test_data_shorthorizon(LocNum, T1, T2, Region, Type, channelname)
          if (dim(forecast_data)[1] < 25) {
            # filename = paste(pathname, method, "_SH_model_", LocNum, ".rds", sep = "")
            # model_fit = readRDS(filename)
            forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
            forecast_usage_rate[forecast_usage_rate < 0] = 0
            print(round(forecast_usage_rate))
            total_usage = sum(forecast_usage_rate, na.rm = T)
            #Delivery_Data$RF_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
            Delivery_Data$RF_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
          } else {
            
            # filename = paste(pathname, method, "_SH_model_", LocNum, ".rds", sep = "")
            # model_fit = readRDS(filename)
            # forecast_data = prepare_test_data_shorthorizon(LocNum, T1, T2, Region, Type, channelname)
            forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
            forecast_usage_rate[forecast_usage_rate < 0] = 0
            forecast_usage_rate = head(forecast_usage_rate, 24)
            SH_usage = sum(forecast_usage_rate, na.rm = T)
            print(paste("SH:", forecast_usage_rate))
            
            # filename = paste(pathname, method, "_LH_model_", LocNum, ".rds", sep = "")
            # model_fit = readRDS(filename)
            #forecast_data = prepare_test_data_longhorizon(LocNum, T1, T2, Region, Type, channelname)
            forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
            forecast_usage_rate[forecast_usage_rate < 0] = 0
            forecast_usage_rate = tail(forecast_usage_rate, -24)
            print(paste("LH:", forecast_usage_rate))
            
            LH_usage = sum(forecast_usage_rate, na.rm = T)
            
            total_usage = SH_usage + LH_usage
            #Delivery_Data$RF_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
            Delivery_Data$RF_predicted_level[g] = Delivery_Data$Level_predicted_from[g] - total_usage
          }
        } else if (RF_version == 1){
          forecast_data = prepare_test_data(LocNum,T1, T2, Region, Type, channelname)
          filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
          model_fit = readRDS(filename)
          forecast_usage_rate = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])] )
          forecast_usage_rate[forecast_usage_rate < 0] = 0
          total_usage = sum(forecast_usage_rate, na.rm = T)
          Delivery_Data$RF_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
        }
        }
        #Delivery_Data$RF_predicted_level[g] = Delivery_Data$Reading_Gals[g] - total_usage
        
        
      }
      
      Delivery_Data$Time_Predicted_from = as.POSIXct(Delivery_Data$Time_Predicted_from, origin = '1970-01-01')
      Delivery_Data = Delivery_Data
      write.csv(Delivery_Data, file = paste0(outpath, "Delivery_data_WnoDR.csv"))
    }, 
    error = function(cond) {
      Delivery_Data = as.data.frame(LocNum)
      Delivery_Data$ReadingDate = NA
      Delivery_Data$ReadingLevel = NA
      Delivery_Data$Reading_Gals = NA
      Delivery_Data$Time_Predicted_from = NA
      Delivery_Data$Time_diff = NA
      Delivery_Data$LBShell_predicted_level = NA
      Delivery_Data$ARIMA_predicted_level = NA
      Delivery_Data$ETS_predicted_level = NA
      Delivery_Data$NN_predicted_level = NA
      #Delivery_Data$RF_LH_predicted_level = NA
      Delivery_Data$RF_predicted_level = NA
      return(Delivery_Data)
    }
    
  )
  Output = Delivery_Data
  # Output = Delivery_Data %>% select(LocNum, ReadingDate, Time_Predicted_from, Time_diff,
  #                                   ReadingLevel, Reading_Gals, LBShell_predicted_level,
  #                                   ARIMA_predicted_level,
  #                                   ETS_predicted_level, NN_predicted_level, 
  #                                   RF_predicted_level)
  # 
  
}



get_LBShell_forecast = function(LocNum, Start_date, End_date, Region, Type, channelname){
  
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  Start_date = as.POSIXct(Start_date, origin = "1970-01-01")
  End_date   = as.POSIXct(End_date, origin = "1970-01-01")
  
  LBFcst_sql = paste("SELECT FHD.[LocNum]
                     ,[ReadingDate]
                     ,[ActualRate]
                     ,[FcstRate]
                     ,CP.[GalsPerInch]
                     FROM ", DB,".[dbo].[FcstHistoryData] FHD
                     INNER JOIN ", DB,".[dbo].[CustomerProfile] CP ON CP.LocNum = FHD.LocNum
                     WHERE FHD.LocNum = '", LocNum,"'
                     and ReadingDate >= '",Start_date - 3600*24*3,"' 
                     and ReadingDate < '", End_date + 3600*24*3,"'
                     ORDER BY ReadingDate")
  channel = RODBC::odbcConnect(channelname)
  LBShell_Data = RODBC::sqlQuery(channel, LBFcst_sql)
  RODBC::odbcClose(channel)
  
  if (dim(LBShell_Data)[1] > 1) {
    #ForecastDataExist = 1
    LBShell_Data$ActualRate_GalperHr = LBShell_Data$ActualRate * 60 # The units in LBShell is gals/min
    LBShell_Data$FcstRate_GalperHr   = LBShell_Data$FcstRate   * 60
    LBFcst_Time = seq.POSIXt(Start_date,End_date, by = "hour")
    LinearForecast = approx(LBShell_Data$ReadingDate,
                            LBShell_Data$FcstRate_GalperHr,
                            LBFcst_Time, method = "constant", rule = 1, f = 0)
    LBShell_forecast = data.frame(LinearForecast$x)
    LBShell_forecast$vslue = LinearForecast$y
    names(LBShell_forecast) = c("Time", "Value")
    
    # Getting the customer profile data
    CusProfile_master_Reading = get_customer_profile(LocNum, Region, Type, channelname)
    names(CusProfile_master_Reading) = c("weekdays", "hours", "Value")
    CusProfile_history        = get_cusprofile_history(LocNum, Start_date, End_date, Region, Type, channelname)
    
    CusProfile_Reading = CusProfile_history %>%
      mutate(weekdays = weekdays(Time),
             hours    = lubridate::hour(Time)) %>%
      left_join(CusProfile_master_Reading, by = c("weekdays", "hours")) %>%
      mutate(Reading  = ifelse(is.na(Reading) == 1, Value, Reading )) %>%
      select(Time, Reading)
    
    LBShell_forecast = inner_join(LBShell_forecast, CusProfile_Reading, by = "Time")
    LBShell_forecast$LBShell = (LBShell_forecast$Value*LBShell_forecast$Reading)/100
    LBShell_forecast = LBShell_forecast %>% select(Time, LBShell)
    
  } else LBShell_forecast = 101
  
  return(LBShell_forecast)
}


get_MAPE = function(Forecast_Data){
  
  
  Forecast_Data = Forecast_Data[is.na(Forecast_Data$Reading_Gals) == 0, ]
  
  # Forecast_Data$ARMIA_AE   = abs((Forecast_Data$Reading_Gals - Forecast_Data$ARIMA_forecasted_level))
  # Forecast_Data$ETS_AE     = abs((Forecast_Data$Reading_Gals - Forecast_Data$ETS_forecasted_level))
  # Forecast_Data$ARIMAX_AE  = abs((Forecast_Data$Reading_Gals - Forecast_Data$ARIMAX_forecasted_level))
  # Forecast_Data$NN_AE      = abs((Forecast_Data$Reading_Gals - Forecast_Data$NN_forecasted_level))
  # Forecast_Data$NNX_AE     = abs((Forecast_Data$Reading_Gals - Forecast_Data$NNX_forecasted_level))
  # Forecast_Data$RF_AE      = abs((Forecast_Data$Reading_Gals - Forecast_Data$RF_forecasted_level))
  # Forecast_Data$LBShell_AE = abs((Forecast_Data$Reading_Gals - Forecast_Data$LBShell_forecasted_level))
  
  Forecast_Data$ARMIA_AE   = abs((Forecast_Data$Reading_Gals - Forecast_Data$ARIMA_predicted_level))
  Forecast_Data$ETS_AE     = abs((Forecast_Data$Reading_Gals - Forecast_Data$ETS_predicted_level))
 # Forecast_Data$ARIMAX_AE  = abs((Forecast_Data$Reading_Gals - Forecast_Data$ARIMAX_forecasted_level))
  Forecast_Data$NN_AE      = abs((Forecast_Data$Reading_Gals - Forecast_Data$NN_predicted_level))
  Forecast_Data$RFLH_AE     = abs((Forecast_Data$Reading_Gals - Forecast_Data$RF_LH_predicted_level))
  Forecast_Data$RFSH_AE      = abs((Forecast_Data$Reading_Gals - Forecast_Data$RF_SH_predicted_level))
  Forecast_Data$RF_AE      = abs(Forecast_Data$Reading_Gals - Forecast_Data$RF_predicted_level)   
  Forecast_Data$LBShell_AE = abs((Forecast_Data$Reading_Gals - Forecast_Data$LBShell_predicted_level))
  
  ind0 = Forecast_Data$Reading_Gals != 0
  ind0[is.na(ind0)] = 0
  #
  Forecast_Data$ARIMA   = ifelse(ind0 == 1, (Forecast_Data$ARMIA_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$ARMIA_AE)
  Forecast_Data$ETS     = ifelse(ind0 == 1, (Forecast_Data$ETS_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$ETS_AE)
  #Forecast_Data$ARIMAX  = ifelse(ind0 == 1, (Forecast_Data$ARIMAX_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$ARIMAX_AE)
  Forecast_Data$NN      = ifelse(ind0 == 1, (Forecast_Data$NN_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$NN_AE)
  #Forecast_Data$NNX     = ifelse(ind0 == 1, (Forecast_Data$NNX_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$NNX_AE)
  Forecast_Data$RFLH      = ifelse(ind0 == 1, (Forecast_Data$RFLH_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$RFLH_AE)
  Forecast_Data$RFSH  = ifelse(ind0 == 1, (Forecast_Data$RFSH_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$RFSH_AE)
  Forecast_Data$RF    = ifelse(ind0 == 1, (Forecast_Data$RF_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$RF_AE)
  Forecast_Data$LBShell = ifelse(ind0 == 1, (Forecast_Data$LBShell_AE/Forecast_Data$Reading_Gals)*100, Forecast_Data$LBShell_AE)
  
  MAPE_Data = Forecast_Data %>%
    group_by(LocNum) %>%
    summarise(ARIMA_MAPE  = mean(ARIMA),
              ETS_MAPE    = mean(ETS),
             # ARIMAX_MAPE = mean(ARIMAX),
              NN_MAPE     = mean(NN),
              #NNX_MAPE    = mean(NNX),
              RFLH_MAPE     = mean(RFLH),
             RFSH_MAPE = mean(RFSH),
             RF_MAPE = mean(RF),
              LBShell_MAPE = mean(LBShell, na.rm = T), 
             num_obs = n())
  
  # #MAPE_Data           = MAPE_Data[is.na(MAPE_Data$ARIMA_MAPE) == 0, ]
  # MAPE_Data$minvalue  = 0
  # MAPE_Data$minmethod = 0
  # for (c in 1:dim(MAPE_Data)[1]){
  #   Data = MAPE_Data[c, 2:7]
  #   MAPE_Data$minvalue[c] = min(Data)
  #   MAPE_Data$minmethod[c] = which.min(Data) + 1
  # }
  MAPE_Data$minvalue  = apply(MAPE_Data[, 2:7], 1, FUN = min, na.rm = T)
  MAPE_Data$minmethod = apply(MAPE_Data[,2:7], 1, FUN = which.min)
  
  #MAPE_Data$RF_MAPE = apply(MAPE_Data[, 5:6],1, min)
  # MAPE_Data = MAPE_Data %>% select(LocNum, ARIMA_MAPE, ETS_MAPE,
  #                                  NN_MAPE, RFSH_MAPE)
  
  Best_Method = c("ARIMA", "ETS", "NN", "RF_LH", "RF_SH", "RF")
  MAPE_Data$best_method = Best_Method[MAPE_Data$minmethod]
  # Output = LN_Data
  # for (y in 1:dim(Output)[1]){
  #   A1 = MAPE_Data %>% filter(LocNum == Output$LocNum[y])
  #   if (dim(A1)[1] > 0){
  #     x = as.numeric(A1[,2:5])
  #     position = match(sort(x), x)
  #   } else {
  #     position = c(4,2,1,3) # Default (RF, ETS, ARIMA, NN)
  #   }
  #   Output$BM1[y] = Best_Method[position[1]]
  #   Output$BM2[y] = Best_Method[position[2]]
  #   Output$BM3[y] = Best_Method[position[3]]
  #   Output$BM4[y] = Best_Method[position[4]]
  # }
# Best_Method_DF = data.frame(minmethod = 1:7, BestMethod = c("ETS", "ARIMA", "ETS", 
#                                                            "ARIMAX", "NN", "NNX", "RF")) 
  
# Best_Method_DF = data.frame(minmethod = 1:8, BestMethod = c("ETS", "ARIMA", "ETS", 
#                                                             "ARIMAX", "NN", "NNX", "RF", "LBShell")) 
# 
# 
#   MAPE_Data = inner_join(MAPE_Data, Best_Method_DF, by = "minmethod")
  return(MAPE_Data)
}



get_cusprofile_history = function(LocNum, Start_date, End_date, Region, Type, channelname) {
  
  Regionname   = get_region(Region)
  TerminalType = get_type(Type)
  DB           = get_database_name(Region)
  
  # Pattern History Data
  channel = RODBC::odbcConnect(channelname)
  
  PH_sql = paste("SELECT [LocNum]
                 ,[DateFrom]
                 ,[DateTo]
                 ,[PatternType]
                 ,[LinearOpPattern]
                 ,[MonShift1]
                 ,[MonShift2]
                 ,[MonShift3]
                 ,[TueShift1]
                 ,[TueShift2]
                 ,[TueShift3]
                 ,[WedShift1]
                 ,[WedShift2]
                 ,[WedShift3]
                 ,[ThuShift1]
                 ,[ThuShift2]
                 ,[ThuShift3]
                 ,[FriShift1]
                 ,[FriShift2]
                 ,[FriShift3]
                 ,[SatShift1]
                 ,[SatShift2]
                 ,[SatShift3]
                 ,[SunShift1]
                 ,[SunShift2]
                 ,[SunShift3]
                 ,[MonShift1StartTime]
                 ,[MonShift1EndTime]
                 ,[MonShift2StartTime]
                 ,[MonShift2EndTime]
                 ,[MonShift3StartTime]
                 ,[MonShift3EndTime]
                 ,[TueShift1StartTime]
                 ,[TueShift1EndTime]
                 ,[TueShift2StartTime]
                 ,[TueShift2EndTime]
                 ,[TueShift3StartTime]
                 ,[TueShift3EndTime]
                 ,[WedShift1StartTime]
                 ,[WedShift1EndTime]
                 ,[WedShift2StartTime]
                 ,[WedShift2EndTime]
                 ,[WedShift3StartTime]
                 ,[WedShift3EndTime]
                 ,[ThuShift1StartTime]
                 ,[ThuShift1EndTime]
                 ,[ThuShift2StartTime]
                 ,[ThuShift2EndTime]
                 ,[ThuShift3StartTime]
                 ,[ThuShift3EndTime]
                 ,[FriShift1StartTime]
                 ,[FriShift1EndTime]
                 ,[FriShift2StartTime]
                 ,[FriShift2EndTime]
                 ,[FriShift3StartTime]
                 ,[FriShift3EndTime]
                 ,[SatShift1StartTime]
                 ,[SatShift1EndTime]
                 ,[SatShift2StartTime]
                 ,[SatShift2EndTime]
                 ,[SatShift3StartTime]
                 ,[SatShift3EndTime]
                 ,[SunShift1StartTime]
                 ,[SunShift1EndTime]
                 ,[SunShift2StartTime]
                 ,[SunShift2EndTime]
                 ,[SunShift3StartTime]
                 ,[SunShift3EndTime]
                 FROM ", DB, ".[dbo].[OpPatternHistory]
                 where DateFrom >= '", Start_date, "' and  DateFrom <= '", End_date, "'
                 AND LocNum = '", LocNum, "'
                 ORDER BY DateFrom")
  
  # PH_sql = readLines("Asia_LB_PatternHistory.txt", warn = F)
  # PH_sql1 = paste(paste(PH_sql[PH_sql!='--'],collapse=' '), "AND LocNum = '", CP_Data$LocNum[count], "'")
  PH_Data = RODBC::sqlQuery(channel, PH_sql)
  RODBC::odbcClose(channel)
  
  Time = seq(from = as.POSIXct(Start_date), to = as.POSIXct(End_date), by = "hour")
  
  CusProfile = as.data.frame(Time)
  CusProfile$Reading = NA
  
  # Check if the data is present 
  PH_Check  = prod(apply(PH_Data, 2, function(x) any(is.na(x))))
  
  if (dim(PH_Data)[1] > 3 & PH_Check != 0) {
    
    for (r in 1:dim(PH_Data)[1]){
      PH1 = PH_Data[r, ]
      T1 = PH1$DateFrom
      T2 = PH1$DateTo
      ind_PH = ((CusProfile$Time >= T1) & (CusProfile$Time <= T2))
      
      # Pattern History = I
      if (PH1$PatternType == "I") {
        CusProfile$Reading[ind_PH] = 0 
      } else {
        CusProfile_A = CusProfile[ind_PH, ]
        
        TimeData  = list()  # Variable to store hourly time element
        ValueData = list() # Variable to store hourly usage element
        PH_Reading   = numeric()
        
        Data = PH1
        ind_V1 = which(names(Data) == "MonShift1")
        ind_D1 = which(names(Data) == "MonShift1StartTime")
        
        for (j in 1:21){
          V1 = as.numeric(Data[ind_V1 + (j-1)])
          D1 = as.POSIXct(as.numeric(Data[ind_D1 + 2*(j-1)]), origin =  "1970-01-01")
          D2 = as.POSIXct(as.numeric(Data[ind_D1 + 1 + 2*(j-1)]), origin =  "1970-01-01")
          Seq = seq(D1, D2, by = "hour")
          start_bias = as.numeric(difftime(D1, as.POSIXct("1970-01-01"), units = "hours"))
          end_bias   = as.numeric(difftime(as.POSIXct("1970-01-02"), D2, units ="hours"))
          #ValueData[j] = list(c(rep(0, start_bias), rep(V1, length(Seq) - 1)))
          if (start_bias >= 0) {
            if (end_bias > 1) {
              ValueData[j] = list(c(rep(0, start_bias), 
                                    rep(V1, length(Seq)),
                                    rep(0, end_bias - 1)))
            } else {
              ValueData[j] = list(c(rep(0, start_bias), 
                                    rep(V1, length(Seq) - 1),
                                    rep(0, end_bias)))
            }
          }
          
        }
        
        x = seq(1,21,3)
        for (v in x){
          VData = rep(0,24)
          
          ind_v1 = unlist(ValueData[v]) != 0
          ind_v2 = unlist(ValueData[v + 1]) != 0
          ind_v3 = unlist(ValueData[v + 2]) != 0
          
          VData[ind_v1] = unlist(ValueData[v])[ind_v1]
          VData[ind_v2] = unlist(ValueData[v + 1])[ind_v2]
          VData[ind_v3] = unlist(ValueData[v + 2])[ind_v3]
          
          PH_Reading = c(PH_Reading,VData)
        }
        
        
        
        PH_TimeSeq = seq(as.POSIXct("2017-06-13 00:00:00"), by = "hour", length.out = 168) # Choose a random day starting at midnight
        PH_TimeSeq = strftime(PH_TimeSeq, format = "%H:%M:%S") # Take out the time part
        
        # I know, I hate this part too!
        Days = c("Monday", "Tuesday", "Wednesday", "Thursday" ,"Friday", "Saturday", "Sunday")
        PH_Days = rep(Days, each = 24)
        PH_Days = paste(PH_Days, PH_TimeSeq) # Add time part to weekdays 
        PH_CusProfile = cbind.data.frame(PH_Reading, PH_Days)
        
        CusProfile_A$Weekday = weekdays(CusProfile_A$Time)
        CusProfile_A$Day_Time = paste(CusProfile_A$Weekday,
                                      strftime(CusProfile_A$Time, format = "%H:%M:%S"))
        
        ind_PH_A = match(CusProfile_A$Day_Time, PH_CusProfile$PH_Days)
        CusProfile_A$Reading = PH_CusProfile$PH_Reading[ind_PH_A]
        
        CusProfile$Reading[CusProfile$Time%in%CusProfile_A$Time] = CusProfile_A$Reading
      }
    }
    return(CusProfile)
  } else {
    return(CusProfile)
  }
  
}


prepare_test_data = function(LocNum, Start_date, End_date, Region, Type, channelname){
  
  
  test_start_date = Start_date
  test_end_date   = End_date
  Start_date      = as.POSIXct(Start_date, origin = "1970-01-01") - 24*60*60*45 # getting past 45 days data since start date
  End_date        = as.POSIXct(End_date, origin = "1970-01-01") + 24*60*60*10   # forward 10 days data after end date
  
  
  Data1 = get_usage_rate_test_data(LocNum, Start_date, End_date, Region, Type, channelname)
  
  if(Data1 != 101) {
    
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 7
    ind_S2 = Data1$Hours >= 8 & Data1$Hours <= 15
    ind_S3 = Data1$Hours >= 16 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    
    # Adding past 3 week usage information 
    Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data_avgWeek = aggregate(UsageRate_Gals_per_Hr ~ Week_Label, data = Data1, sum, na.rm = T)
    colnames(Data_avgWeek)[2] = "WeekSum"
    ind_avgweek = match(Data1$Week_Label, Data_avgWeek$Week_Label)
    Data1$WeekSum = Data_avgWeek$WeekSum[ind_avgweek]
    
    cl = length(Data1$UsageRate_Gals_per_Hr)
    LastWeekEnd = cl-168
    Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    
    Week2St = 2*168+1
    Week2End = cl-2*168
    Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    
    Week3St = 3*168+1
    Week3End = cl-3*168
    Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    
    # Average Shift usage rate
    Data_Shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, data = Data1, mean, na.rm = T)
    colnames(Data_Shift)[2] = "Avg_Shift_Usage"
    ind_Shift = match(Data1$Shift, Data_Shift$Shift)
    Data1$Avg_Shift_Usage = Data_Shift$Avg_Shift_Usage[ind_Shift]
    
    # Average Daily usage rate
    Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    colnames(Data_Day)[2] = "Avg_Day_Usage"
    ind_Day = match(Data1$Days, Data_Day$Days)
    Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    
    # Average Monthly usage rate
    Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    colnames(Data_Month)[2] = "Avg_Month_Usage"
    ind_month = match(Data1$Month, Data_Month$Month)
    Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    
    
    
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    #if (training == 0){
    ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    Data_F = Data1[ind_test, ]
    # } else  {
    #   ind_na = is.na(Data1$Last3WeekSum)
    #   Data_F = Data1[!ind_na, ]
    # }
    
    Data_F = dplyr::select(Data_F, Time, UsageRate_Gals_per_Hr, Days, Shift, Hours,
                           Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
                           Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    
  } else return(101)
  
  # # --------------------------------------------------------------------------
  # # --------------------------------------------------------------------------
  # # Hourly Prediction 
  # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
  # Xvar_Data = Data_F
  # 
  # if (methodclass == 1){ return(Timeseries_Data)}
  # else if (methodclass == 2) { return(Xvar_Data)}
  # else print("Provide a method class")
}

prepare_training_data = function( LocNum, Start_date, End_date, Region, Type, channelname){
  
  # if (training == 0){
  #   test_start_date = Start_date
  #   test_end_date   = End_date
  #   Start_date = as.POSIXct(Start_date, origin = "1970-01-01") - 24*60*60*45 # getting past 45 days data
  #   End_date   = as.POSIXct(End_date, origin = "1970-01-01") + 24*60*60*10 # forward 10 days data 
  # }
  
  Data1 = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname)
  
  if (Data1 != 101){ # Past data exist
    # --------------------------------------------------------
    # Adding features to the data
    # -------------------------------------------------------
    # Add shift information 
    Data1$Days  = weekdays(Data1$Time)
    Data1$Hours = lubridate::hour(Data1$Time)
    Data1$Month = lubridate::month(Data1$Time)
    
    ind_S1 = Data1$Hours >= 0 & Data1$Hours <= 7
    ind_S2 = Data1$Hours >= 8 & Data1$Hours <= 15
    ind_S3 = Data1$Hours >= 16 & Data1$Hours <= 23
    
    Data1$Shift[ind_S1] = "Shift1"
    Data1$Shift[ind_S2] = "Shift2"
    Data1$Shift[ind_S3] = "Shift3"
    
    # Adding past 3 week usage information 
    Data1$Week_Label = ceiling(c(1:dim(Data1)[1])/168) # number of Week label 
    Data_avgWeek = aggregate(UsageRate_Gals_per_Hr ~ Week_Label, data = Data1, sum, na.rm = T)
    colnames(Data_avgWeek)[2] = "WeekSum"
    ind_avgweek = match(Data1$Week_Label, Data_avgWeek$Week_Label)
    Data1$WeekSum = Data_avgWeek$WeekSum[ind_avgweek]
    
    cl = length(Data1$UsageRate_Gals_per_Hr)
    LastWeekEnd = cl-168
    Data1$LastWeekSum[169:cl] = Data1$WeekSum[1:LastWeekEnd]
    
    Week2St = 2*168+1
    Week2End = cl-2*168
    Data1$Last2WeekSum[Week2St:cl] = Data1$WeekSum[1:Week2End]
    
    Week3St = 3*168+1
    Week3End = cl-3*168
    Data1$Last3WeekSum[Week3St:cl] = Data1$WeekSum[1:Week3End]
    
    # Average Shift usage rate
    Data_Shift = aggregate(UsageRate_Gals_per_Hr ~ Shift, data = Data1, mean, na.rm = T)
    colnames(Data_Shift)[2] = "Avg_Shift_Usage"
    ind_Shift = match(Data1$Shift, Data_Shift$Shift)
    Data1$Avg_Shift_Usage = Data_Shift$Avg_Shift_Usage[ind_Shift]
    
    # Average Daily usage rate
    Data_Day = aggregate(UsageRate_Gals_per_Hr ~ Days, data = Data1, mean, na.rm = T)
    colnames(Data_Day)[2] = "Avg_Day_Usage"
    ind_Day = match(Data1$Days, Data_Day$Days)
    Data1$Avg_Day_Usage = Data_Day$Avg_Day_Usage[ind_Day]
    
    # Average Monthly usage rate
    Data_Month = aggregate(UsageRate_Gals_per_Hr ~ Month, data = Data1, mean, na.rm = T)
    colnames(Data_Month)[2] = "Avg_Month_Usage"
    ind_month = match(Data1$Month, Data_Month$Month)
    Data1$Avg_Month_Usage = Data_Month$Avg_Month_Usage[ind_month]
    
    
    
    # ============================================================================
    # Forecast Data
    # ============================================================================
    
    # Removing first 3 weeks (as they do not have complete information)
    # if (training == 0){
    #   ind_test = Data1$Time >= test_start_date & Data1$Time <= test_end_date
    #   Data_F = Data1[ind_test, ]
    # } else  {
    ind_na = is.na(Data1$Last3WeekSum)
    Data_F = Data1[!ind_na, ]
    #}
    
    Data_F = dplyr::select(Data_F, UsageRate_Gals_per_Hr, Days, Shift, Hours,
                           Month, LastWeekSum,Last2WeekSum, Last3WeekSum,
                           Avg_Shift_Usage, Avg_Day_Usage, Avg_Month_Usage)
    
    Data_F$Hours = as.factor(Data_F$Hours)
    Data_F$Days  = as.factor(Data_F$Days)
    Data_F$Month = as.factor(Data_F$Month)
    Data_F$Shift = as.factor(Data_F$Shift)
    
    # Data_F = Data_F[order(Data_F$Time), ]
    # Data_F = Data_F[c(2:dim(Data_F)[2])]
    
    levels(Data_F$Days)  = c(5,1,6,7,4,2,3)
    levels(Data_F$Shift) = c(1,2,3)
    levels(Data_F$Hours) = 0:23
    levels(Data_F$Month) = 1:12
    
    return(Data_F)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Hourly Prediction 
    # Timeseries_Data = ts(Data_F$UsageRate_Gals_per_Hr)
    # Xvar_Data = Data_F
    # 
    # if (methodclass == 1){ return(Timeseries_Data)}
    # else if (methodclass == 2) { return(Xvar_Data)}
    # else print("Provide a method class")
  } else {
    return(101)
  }
}

bestmethod_stat = function(Locnum, bestmethod_filename){
  bestmethod_dta = read.csv(file = paste(bestmethod_filename, "Best_Method_table.csv", sep = "_"), stringsAsFactors = F)
  output = bestmethod_dta %>% filter(LocNum == Locnum)
}

runout_forecast = function(LocNum, Start_date, End_date, Region, Type, channelname, pathname, bestmethod_filename) {
  
  bm_dta = bestmethod_stat(LocNum, bestmethod_filename) # best method data
  method = bm_dta$best_method
  
  filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
  model_fit = readRDS(filename)

  actual_data = get_actual_readings(LocNum, Start_date - 168*60*60, Start_date, Region, Type, channelname)   
  if (dim(actual_data)[1] > 0) {
    nondelivery_data = actual_data %>% filter(ReadingType != 2, ReadingType != 3)
    
    runout_level = actual_data$RunoutGals[1]
    if (dim(nondelivery_data)[1] > 0) {
      last_available_time = tail(nondelivery_data$ReadingDate, 1)
      last_available_reading = tail(nondelivery_data$Reading_Gals, 1)
    } else {
      last_available_time = tail(actual_data$ReadingDate, 1)
      last_available_reading = tail(actual_data$Reading_Gals, 1)
    }
  
    forecast_data = prepare_test_data_shorthorizon(LocNum, last_available_time, End_date, Region, Type, channelname)
  
    if (forecast_data != 101) {
      pred_horizon = dim(forecast_data)[1]
      
      if (method == "ARIMA" | method == "ETS" | method == "NN")  {
        model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
        model_forecast  = model_forecast$mean
        
      } else if (method == "ARIMAX" | (method == "NNX"))  {
        xVars           = data.matrix(forecast_data[, c(3: dim(forecast_data)[2])])
        model_forecast  = forecast::forecast(model_fit, xreg = xVars, h = pred_horizon)
        model_forecast  = model_forecast$mean
        
      } else if (method == "RF_LH" | method == "RF_SH") {
        model_forecast = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])])
        model_forecast[model_forecast < 0] = 0
        
      } else if (method == "RF"){
        forecast_data = prepare_test_data(LocNum, Start_date, End_date, Region, Type, channelname)
        model_forecast = predict(model_fit, forecast_data[, c(2:dim(forecast_data)[2])])
        model_forecast[model_forecast < 0] = 0
      }
    
      
      # forecast usage rate output
      forecast_data$predicted_usage_rate = model_forecast
      forecast_data$predicted_level = last_available_reading - cumsum(model_forecast)
      forecast_data$runout_chance = forecast_data$predicted_level - runout_level
      ind_ro = forecast_data$runout_chance <= 0 
      
      forecast_output = forecast_data %>% select(Time, predicted_usage_rate) %>%
        filter(Time >= Start_date)
      outfilepath = "\\\\home7\\Forecast\\Asia_MB_Forecast\\Output\\UsageRate\\"
      starttime = paste(as.Date(Start_date), strftime(Start_date, format = "%H-%M-%S"), sep = "_")
      write.csv(forecast_output, file = paste0(outfilepath, starttime, "_",LocNum, "_usage_rate.csv"), row.names = F)
      
      # Run out level output 
      predicted_level = last_available_reading - sum(model_forecast, na.rm = T)
      predicted_level_24hr = last_available_reading - sum(head(model_forecast, -144), na.rm = T)
      
      runout_dta = as.data.frame(LocNum)
      runout_dta$current_time = Start_date
      runout_dta$last_available_reading_time = last_available_time
      runout_dta$last_available_reading_KG = round(last_available_reading)
      runout_dta$runout_level_KG = runout_level
      runout_dta$level_prediction_24hr  = round(predicted_level_24hr)
      runout_dta$level_prediction_1week = round(predicted_level)
      
      # Computing probabilty of runout in next 24 hrs
      err_factor = ifelse(bm_dta$num_obs > 5, 1, 5/bm_dta$num_obs)
      runout_prob_24hr = (bm_dta$minvalue/(predicted_level_24hr - runout_level))*err_factor*100
      runout_prob_24hr = round(ifelse(runout_prob_24hr >=0, runout_prob_24hr, 100))
      runout_prob_24hr = round(ifelse(runout_prob_24hr >=100, 100, runout_prob_24hr))
      
      if (predicted_level > runout_level){
        runout_dta$runout_time_prediction = NA
        runout_dta$runout_forecast = 'greater than a week'
        runout_dta$runout_prob_24hr = runout_prob_24hr
      } else {
        runout_dta$runout_time_prediction =  forecast_data$Time[head(which(ind_ro == 1),1)]
        runout_days = difftime(runout_dta$runout_time, Start_date, units = 'days')
        runout_dta$runout_forecast = paste(round(as.numeric(runout_days) ,2), "days", sep = " ")
        runout_dta$runout_prob_24hr = runout_prob_24hr
      }
      
      
      runout_dta = runout_dta
    }
  }
    
}


houroftheday_data = function(dta){
  dta = get_usage_rate_data(LocNum, Start_date, End_date, Region, Type, channelname, driverreadingflag, remvarianceflag)
  dta$month = lubridate::month(dta$Time)
  dta$hours = lubridate::hour(dta$Time)
  dta$weekdays = weekdays(dta$Time)
  
  hr_dta = dta %>% group_by(hours, weekdays) %>%
    summarise(usage_data = mean(UsageRate_Gals_per_Hr, na.rm = T))
  
  cust_pattern = get_customer_profile(LocNum, Region, Type, channelname)
  
  dummy_week_seq = data.frame(seq(as.POSIXct("2017-12-18"), by = "hour", length.out = 168)) # starting Monday
  names(dummy_week_seq) = 'Time'
  dummy_week_seq$weekdays = weekdays(dummy_week_seq$Time)
  dummy_week_seq$hours = lubridate::hour(dummy_week_seq$Time)
  
  cust_pattern = inner_join(dummy_week_seq, cust_pattern, by = c('weekdays', 'hours'))
  out_dta      = inner_join(cust_pattern, hr_dta, by = c('weekdays', 'hours'))
  
  outpath    =  "\\\\home7\\Forecast\\LBShell_Forecast\\Output\\"
  write.csv(out_dta, file = paste0(outpath, LocNum, "_", Region,'_houroftheday_data.csv'), row.names = F)
  
  out_dta %>%
    ggplot(aes(x = Time, y = usage_data)) +
    geom_bar(stat = 'Identity')
}

{
# get_forecast_PH = function(LocNum, Start_date, End_date, Region, Type, channelname, pathname, method) {
#   
#   #forecast_data = prepare_test_data(LocNum, Start_date, End_date, Region, Type, channelname)
#   Actual_data = get_actual_readings(LocNum, Start_date, End_date, Region, Type, channelname)
#   pred_horizon = 60
#   method = "ARIMA"
#   filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
#   model_fit = readRDS(filename)
#   k = 0
#   forecast_usage_rate =  foreach(i = 1:dim(Actual_data)[1], .combine = 'rbind', .packages = c("RODBC", "dplyr", "forecast")) %do% {
#     model_forecast = forecast::forecast(model_fit, h = pred_horizon)
#     k = k + 1
#     if (k > 4){
#       Startdate = Actual_data$ReadingDate[i-k+1]
#       Enddate = Actual_data$ReadingDate[i]
#       new_usage = get_usage_rate_test_data(LocNum, Startdate, Enddate, Region, Type, channelname)
#       usage_data = ts(new_usage$UsageRate_Gals_per_Hr)
#       model_fit = forecast::Arima(usage_data, model = model_fit)
#       k = 0
#     }
#     forecast_usage_rate = model_forecast$mean
#   }
#   
#   Actual_data = cbind.data.frame(Actual_data, forecast_usage_rate)
#   
#   
#   
#   return(Actual_data)
# }

# Latest get forecast



  # 
  # get_forecast_level = function(LocNum, Start_date, End_date, Region, Type, channelname, pathname, Method){
  #   
  #   #cus_stat_data = get_reading_statistics(LocNum, Start_date, channelname)
  #   
  #   
  #   
  #   Actual_data = get_actual_readings(LocNum, Start_date, End_date, Region, Type, channelname)
  #   #Actual_data = Actual_data %>% filter(ReadingType == 2 | ReadingType == 3) # Only for delivery forecast
  #   
  #   if (dim(Actual_data)[1] > 0 & dim(Actual_data)[2] > 4){ 
  #     # Checking if actual data is present or not and there is a level information 
  #     # And if segment data available for the forecast dates 
  #     
  #     # Getting the forecast usage rate 
  #     Forecast_data    = get_forecast(LocNum, Start_date, End_date, Region, Type, channelname, pathname, Method)
  #     LBShell_forecast = get_LBShell_forecast(LocNum, Start_date, End_date, Region, Type, channelname)
  #     
  #     
  #     if (any(is.na(Forecast_data)) == 0) { # Checking if the forecast is present or not 
  #       
  #       #if (dim(Actual_data)[2] > 4) { # No segment data available for the forecast dates 
  #       
  #       Actual_data$hourly_time = as.POSIXct(round(Actual_data$ReadingDate, units = "hours"))
  #       # Time differenc between readings
  #       Actual_data$Days_btw_readings = round(c(0, diff(Actual_data$ReadingDate)/(24*60)), 2)
  #       # Time difference between rounded time and reading time
  #       Actual_data$Time_diff = difftime(Actual_data$hourly_time, Actual_data$ReadingDate, units = "mins")
  #       
  #       if (LBShell_forecast != 101){
  #         Method = c(Method, "LBShell")
  #         Forecast_data = inner_join(Forecast_data, LBShell_forecast, by = "Time")
  #       }
  #       
  #       
  #       for (m_ind in 1:length(Method)) {
  #         Actual_data$forecasted_level = 0
  #         
  #         forecast_usage_rate = Forecast_data[, m_ind + 2] # NEED TO BE CAREFUL AS STRUCTURE CAN CHANGE
  #         # Computing the forecasted level
  #         for (i in 1: (dim(Actual_data)[1]-1)){
  #           T1 = Actual_data$hourly_time[i]
  #           T2 = Actual_data$hourly_time[i+1]
  #           ind_time = Forecast_data$Time >= T1 & Forecast_data$Time <= T2
  #           usage = sum(forecast_usage_rate[ind_time]) + 
  #             as.numeric(Actual_data$Time_diff[i]/60)*forecast_usage_rate[Forecast_data$Time == T1] -
  #             as.numeric(Actual_data$Time_diff[i+1]/60)*forecast_usage_rate[Forecast_data$Time == T2]
  #           Days_btw_readings = Actual_data$ReadingDate[i] 
  #           Actual_data$forecasted_level[i+1] = Actual_data$Reading_Gals[i] - usage
  #         }
  #         
  #         Actual_data$forecasted_level[Actual_data$forecasted_level < 0] = 0
  #         method = Method[m_ind]
  #         names(Actual_data)[names(Actual_data) == 'forecasted_level'] <- paste(method, "_forecasted_level", sep = "")
  #       }
  #       if (LBShell_forecast == 101) {Actual_data$LBShell_forecasted_level = NA}
  #       # Removing the first record and record with reading type = 3
  #       Actual_data = Actual_data[2:dim(Actual_data)[1], ]
  #       Actual_data = Actual_data %>%
  #         filter(ReadingType != 3)
  #       
  #       # Absolute percentage error 
  #       # Actual_data$forecast_level_APE = (abs(Actual_data$Reading_Gals - Actual_data$forecasted_level)/Actual_data$Reading_Gals)*100
  #       Actual_data = Actual_data[, -c(7,8,9)] # Removing unwanted column
  #       
  #       # } else { # No segment data available for the forecast dates 
  #       #   Actual_data$Reading_Gals = NA
  #       #   Actual_data$Actual_Delivered_Amount = NA
  #       #   Actual_data$forecasted_level = NA
  #       #   Actual_data$forecast_level_APE = NA
  #       # }
  #       Output = Actual_data
  #       #return(Output)
  #     } else {
  #       
  #       # If no forecast data exist
  #       Actual_data$NN_forecasted_level = NA
  #       Actual_data$ARIMA_forecasted_level = NA
  #       Actual_data$ETS_forecasted_level = NA
  #       Actual_data$RF_forecasted_level = NA
  #       Actual_data$ARIMAX_forecasted_level = NA
  #       Actual_data$NNX_forecasted_level = NA
  #       Actual_data$LBShell_forecasted_level = NA
  #       #Actual_data$forecast_level_APE = NA
  #       Output = Actual_data
  #       #return(Output)
  #     } 
  #     
  #   } else {
  #     
  #     # If no actual data exist 
  #     Output = data.frame(LocNum)
  #     Output$ReadingDate  = NA
  #     Output$ReadingLevel = NA
  #     Output$ReadingType = NA
  #     Output$Reading_Gals = NA
  #     Output$Actual_Delivered_Amount = NA
  #     Output$RF_forecasted_level = NA
  #     Output$ARIMAX_forecasted_level = NA
  #     Output$NNX_forecasted_level = NA
  #     Output$NN_forecasted_level = NA
  #     Output$ARIMA_forecasted_level = NA
  #     Output$ETS_forecasted_level = NA
  #     Output$LBShell_forecasted_level = NA
  #     #Output$forecast_level_APE = NA
  #   }
  #   
  #   
  #   return(Output)
  # }  
  
# get_forecast = function(LocNum, Start_date, End_date, Region, Type, channelname, method) {
# 
#   pathname = "\\\\home7\\DecisionScience\\Anshu Gupta\\LBShell Forecast\\model\\"
#   filename = paste(pathname, method, "_model_", LocNum, ".rds", sep = "")
#   if (file.exists(filename) == 1){
#     # Using existing model
#     model_fit = readRDS(filename)
#     if (typeof(model_fit) == "character") {
# 
#       print(model_fit) # No past data avaulable for model
#       Output = data.frame(LocNum)
#       Output$Time = Start_date
#       Output$Method = method
#       Output$forecast_usage_rate = NA
#       Output$Comment = model_fit
# 
#     } else {
# 
#       forecast_data = prepare_test_data(LocNum, Start_date, End_date, Region, Type, channelname)
#       # If data is available to forecast (101 is used as a flag to show no data)
#       if (forecast_data != 101) {
#         pred_horizon = dim(forecast_data)[1]
# 
#         if (method == "ARIMA" | method == "ETS" | method == "NN")  {
#           # Actual_data = get_actual_readings(LocNum, Start_date, End_date, Region, Type, channelname)
#           # foreach(p = 1:dim(Actual_data)[1])
#           #
#           model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
#         } else if (method == "ARIMAX" | (method == "NNX"))  {
#           xVars           = data.matrix(forecast_data[, c(3: dim(forecast_data)[2])])
#           model_forecast  = forecast::forecast(model_fit, xreg = xVars, h = pred_horizon)
#           model_forecast  = model_forecast$mean
#           # } else if (method == "ETS") {
#           #   # Exponential smooting
#           #   model_forecast  = forecast::forecast(model_fit, h = pred_horizon)
#           # } else if (method == "NN") {
#           #   # Neural network
#           #   model_forecast      = forecast::forecast(model_fit, h = pred_horizon)
#           # } else if (method == "NNX") {
#           #   # Neural network with x variable
#           #   xVars           = data.matrix(forecast_data[, c(2: dim(forecast_data)[2])])
#           #   model_fit       = forecast::nnetar(ts(forecast_data[, 1]), xreg = xVars, MaxNWts = 3000)
#         } else if (method == "RF") {
#           # levels(forecast_data$Days) = 1:7
#           # levels(forecast_data$Shift) = 1:3
#           # levels(forecast_data$Hours) = 0:23
#           # levels(forecast_data$Month) = 1:12
#           model_forecast = predict(model_fit, forecast_data[, c(3:dim(forecast_data)[2])] )
#         }
#         forecast_data$LocNum = LocNum
#         forecast_data$Method = method
#         forecast_data$forecast_usage_rate = model_forecast
#         forecast_data$Comment = ""
# 
#         Output = forecast_data %>%
#           select(Time, LocNum, Method, forecast_usage_rate, Comment)
# 
#       } else {
# 
#         Output = data.frame(LocNum)
#         Output$Time = Start_date
#         Output$Method = method
#         Output$forecast_usage_rate = NA
#         Output$Comment = paste("No current data for", LocNum, "between", Start_date, End_date)
# 
#       }
# 
#     }
#     #return(101)
#   } else {
#     # Need to create a new model
#     Output = data.frame(LocNum)
#     Output$Time = Start_date
#     Output$Method = method
#     Output$forecast_usage_rate = NA
#     Output$Comment = "No current model in directory - fit new model"
#     #return(101)
#     # End_date = Sys.Date()
#     # Start_date = seq(as.Date(End_date), length = 2, by = "-12 months")[2] # 12 months back from today
#     # model_fit = train_forecast_model(LocNum, Start_date, End_date, Region, Type, channelname, method)
#   }
# 
#   return(Output)
# }
  
  
  # get_usage_rate_data = function(LocNum, Start_date, End_date, Region, Type, channelname)  {
  #   
  #   # packages 
  #   # require(RODBC)
  #   # require(dplyr)
  #   Regionname   = get_region(Region)
  #   TerminalType = get_type(Type)
  #   DB           = get_database_name(Region)
  #   
  #   # --------------------------------------------------------------------
  #   
  #   
  #   # Getting the customer profile data 
  #   # CP_sql = paste("SELECT [LocNum]
  #   #                          ,[ProductAbbrev]
  #   #                          ,[GalsPerInch]
  #   #                          FROM ", DB,".[dbo].[CustomerProfile] 
  #   #                          WHERE Locnum = '", LocNum, "'")
  #   #CP_sql1 = paste(paste(CP_sql[CP_sql!='--'],collapse = ' '), "WHERE Locnum = '", LocNum, "'")
  #   
  #   
  #   # Telemetry Reading Data
  #   # Reading data from now to 6 months back
  #   # TR_end_time   = Sys.time() # Current time 
  #   # TR_start_time = seq(as.Date(TR_end_time), length = 2, by = "-6 months")[2] # 6 months back from today
  #   # TR_start_time = as.POSIXct(TR_start_time)
  #   TR_start_time = as.POSIXct(Start_date, origin = "1970-01-01")
  #   TR_end_time   = as.POSIXct(End_date, origin = "1970-01-01")
  #  
  #   # 
  #   TR_sql = paste("SELECT R.[LocNum]
  #                  ,[ReadingDate]
  #                  ,CASE WHEN [ReadingUOM] = 1 THEN [ReadingLevel] / [GalsPerInch] ELSE [ReadingLevel] END AS ReadingLevel
  #                  ,[ReadingType]
  #                  ,CASE WHEN [ReadingUOM] = 1 THEN [ReadingLevel]  ELSE [ReadingLevel] * [GalsPerInch] END AS Reading_Gals
  #                  FROM ", DB,".[dbo].[Readings] R
  #                  JOIN ", DB,".[dbo].[CustomerProfile] CP
  #                  ON R.LocNum = CP.LocNum
  #                  WHERE ReadingDate >= '", TR_start_time, "'",
  #                  "AND ReadingDate <'", TR_end_time, "'",
  #                  "AND (ReadingLevel > 0 or (readinglevel = 0 and readingtype = 2))
  #                  AND ReadingUOM != 2
  #                  AND R.Locnum = '", LocNum, "'",
  #                  "AND ReadingStatus != 2
  #                  ORDER BY ReadingDate")
  # 
  #   
  #   channel = RODBC::odbcConnect(channelname)
  #   #CP_Data = RODBC::sqlQuery(channel, CP_sql)
  #   TR_Data = RODBC::sqlQuery(channel, TR_sql, as.is = T)
  #   RODBC::odbcClose(channel)
  #   
  #   
  #   
  #   # ------------------------------------------------------------------------- 
  #   # Running only if at least 5 telemetry data exist for the customer
  #   if(dim(TR_Data)[1] > 3) {
  #     # Adding reading in gals
  #     #TR_Data$Reading_Gals = TR_Data$ReadingLevel*CP_Data$GalsPerInch
  #     TR_Data$ReadingDate  = as.POSIXct(TR_Data$ReadingDate, tz = "EST")
  #     TR_Data$ReadingLevel = as.numeric(TR_Data$ReadingLevel)
  #     TR_Data$Reading_Gals = as.numeric(TR_Data$Reading_Gals)
  #     # Getting the customer profile data
  #     CusProfile_Reading = get_customer_profile(LocNum, Region, Type, channelname)
  #     
  #     
  #     # Creating the time sequence for the length of dataset (exclude timeset until first data is found)
  #     last_data_time = tail(TR_Data$ReadingDate, 1)
  #     first_data_time = round(head(TR_Data$ReadingDate, 1), units = "hours") - 60*60 # first hour closest to the first reading
  #     data_length_hours = round(as.numeric(difftime(last_data_time, first_data_time, units = "hours")))
  #     Time_Seq = seq(first_data_time, by = "hour", length.out = data_length_hours)
  #     # data_length_hours = round(as.numeric(difftime(TR_end_time, TR_start_time, units = "hours")))
  #     # Time_Seq = seq(TR_start_time, by = "hour", length.out = data_length_hours)
  #     
  #     #Reading = head(rep(Reading, 105), 17545)
  #     Time_Seq = as.data.frame(Time_Seq)
  #     names(Time_Seq) = "Time"
  #     
  #     CusProfile = Time_Seq %>%
  #       mutate(weekdays = weekdays(Time),
  #              hours    = lubridate::hour(Time),
  #              LocNum   = LocNum) %>%
  #       left_join(CusProfile_Reading, by = c("weekdays", "hours"))
  #     
  #     
  #     # ------------------------------------------------------------------------------
  #     # Telemetry Reading Data
  #     # ------------------------------------------------------------------------------
  #     
  #     TR_Data = TR_Data[order(TR_Data$ReadingDate), ]
  #     LinearApprox = approx(TR_Data$ReadingDate,
  #                           TR_Data$Reading_Gals,
  #                           Time_Seq$Time, method = "linear", rule = 2)
  #     
  #     CusProfile$LinearReadings = LinearApprox$y
  #     CusProfile$LinearUsage = round(-c(0,diff(CusProfile$LinearReadings)),3)
  #     CusProfile$LinearUsage[CusProfile$LinearUsage < 0] = NA
  #     
  #     #CusProfile$LinearUsage[is.na(CusProfile$LinearUsage) == 1] = median(CusProfile$LinearUsage, na.rm = T)
  #     CusProfile$LinearUsage = zoo::na.locf(CusProfile$LinearUsage)
  #     
  #     # Removing outliers | replacing the value with mean (double pass)
  #     # n.pass = 0
  #     # for (i in 1:n.pass) {
  #     #   ind_nonzero = CusProfile$LinearUsage > 0
  #     #   HighVal = mean(CusProfile$LinearUsage[ind_nonzero]) + 3*sd(CusProfile$LinearUsage[ind_nonzero])
  #     #   CusProfile$LinearUsage[CusProfile$LinearUsage > HighVal] = median(CusProfile$LinearUsage, na.rm = T)
  #     # }
  #     
  #     # Converting hourly readings to UsageRate (Gals/Hr) 
  #     # Using the customer profile data between each reading
  #     
  #     Reading_Time = unique(as.POSIXct(TR_Data$ReadingDate), na.rm = T)
  #     Reading_Time = Reading_Time[is.na(Reading_Time) == 0]
  #     for (i in 2:length(Reading_Time)){
  #       ind_BR = (CusProfile$Time > Reading_Time[i-1] & CusProfile$Time <= Reading_Time[i])
  #       Num_Readings = sum(CusProfile$Reading[ind_BR])
  #       if ((Num_Readings/100 == sum(ind_BR)) | (Num_Readings == 0)) {
  #         CusProfile$UsageRate_Gals_per_Hr[ind_BR] = CusProfile$LinearUsage[ind_BR]
  #       } else {
  #         CusProfile$UsageRate_Gals_per_Hr[ind_BR] = (sum(CusProfile$LinearUsage[ind_BR]))*
  #           (CusProfile$Reading[ind_BR]/Num_Readings)
  #       }
  #     }
  #     ind_na = is.na(CusProfile$UsageRate_Gals_per_Hr)
  #     CusProfile$UsageRate_Gals_per_Hr[ind_na] = CusProfile$LinearUsage[ind_na]
  #     CusProfile$UsageRate_Gals_per_Hr = round(CusProfile$UsageRate_Gals_per_Hr, 3)
  #     
  #     CusProfile = CusProfile %>%
  #       select(LocNum, Time, Reading, LinearReadings, LinearUsage, UsageRate_Gals_per_Hr)
  #     
  #     names(CusProfile) = c("LocNum", "Time", "CustomerPattern", "LinearUsage_Gals",
  #                           "LinearUsage_Gals_per_Hr", "UsageRate_Gals_per_Hr")
  #     return(CusProfile)
  #     
  #   } else {
  #     # print(paste("No Telemetry data for LocNum = ", LocNum))
  #     # break
  #     
  #     CusProfile = 101
  #   }
  # } 
}