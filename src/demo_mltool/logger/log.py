# -*- coding: utf-8 -*-

__author__ = "wangzhefeng"


# logger：提供日志接口，供应用代码使用
    #- logger最常用的操作有两类：配置和发送日志消息。可以通过logging.getLogger(name)获取logger对象，如果不指定name则返回root对象，多次使用相同的name调用getLogger方法返回同一个logger对象
# handler：将日志记录(log record)发送到合适的目的地(destination)，比如文件，socket等
    #- 一个logger对象可以通过addHandler方法添加到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示
# filter：提供一种优雅的方式决定一个日志记录是否发送到handler
# formatter：指定日志记录输出的具体格式
    #- formatter的构造方法需要两个参数：消息的格式字符串和日期字符串，这两个参数都是可选的


# 日志级别
# NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL


# logging用法解析
# 1. 初始化 
    # logger = logging.getLogger("endlesscode")
    # getLogger()方法后面最好加上所要日志记录的模块名字，后面的日志格式中的%(name)s对应的是这里的模块名字
# 2. 设置级别 
    # logger.setLevel(logging.DEBUG)
    # Logging中有 NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL 这几种级别，日志会记录设置级别以上的日志
# 3. Handler
    # 常用的是StreamHandler和FileHandler，windows下你可以简单理解为一个是console和文件日志，一个打印在CMD窗口上，一个记录在一个文件上
# 4. formatter
    #定义了最终log信息的顺序,结构和内容，比如这样的格式 '[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'，
    # %(name)s 				Logger的名字
    # %(levelname)s 		文本形式的日志级别
    # %(message)s 			用户输出的消息
    # %(asctime)s 			字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
    # %(levelno)s 			数字形式的日志级别
    # %(pathname)s 			调用日志输出函数的模块的完整路径名，可能没有
    # %(filename)s 			调用日志输出函数的模块的文件名
    # %(module)s 			调用日志输出函数的模块名
    # %(funcName)s 			调用日志输出函数的函数名
    # %(lineno)d 			调用日志输出函数的语句所在的代码行
    # %(created)f 			当前时间，用UNIX标准的表示时间的浮 点数表示
    # %(relativeCreated)d 	输出日志信息时的，自Logger创建以 来的毫秒数
    # %(thread)d 			线程ID。可能没有
    # %(threadName)s 		线程名。可能没有
    # %(process)d 			进程ID。可能没有
# 5. 记录 
    # 使用object.debug(message)来记录日志


# ********************************************************
import logging

logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 建立一个filehandle来把日志记录在文件里, 级别为debug以上
fh = logging.FileHandler("spam.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)

# 将相应的handler添加在logger对象中
logger.addHandler(fh)
logger.addHandler(ch)

# 开始打印日志
logger.debug("debug message")
logger.info("info message")
logger.warn("warn message")
logger.error("error message")
logger.critical("cirtical message")

