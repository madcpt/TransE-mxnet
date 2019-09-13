def Log(default_level, log_path, live_stream):
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(default_level)
    formatter = logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    if log_path is not None:
        import time
        time_tag = time.strftime("%Y-%a-%b-%d-%H-%M-%S", time.localtime())
        file_path = 'log/{}-{}.log'.format(log_path, time_tag)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if live_stream is not None and live_stream == True:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
    
    # self.__logger = logger
    # self.__log_path = log_path
    # self.__live_stream = live_stream
    return logger

    
    # def info(self, message):
    #     self.__logger.info(message)
        
    # def debug(self, message):
    #     self.__logger.debug(message)
        
    # def warning(self, message):
    #     self.__logger.warning(message)
        
    # def error(self, message):
    #     self.__logger.error(message)
        
    # def critical(self, message):
    #     self.__logger.critical(message)
