conn = DBI::dbConnect(odbc::odbc(),
                      Driver = "SQL Server",
                      Server = "WANGZF-PC",
                      Database = "tinker",
                      UID = "tinker.wang",
                      PWD = "alvin123",
                      Port = 1433)
