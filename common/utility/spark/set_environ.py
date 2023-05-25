# NOTE: INSTALL SPARK ON WINDOWS
# import os
# import sys
#
# from pyspark.sql import SparkSession
#
#
# def get_session(app_name: str, max_cores: int = 12) -> SparkSession:
# 	os.environ['PYSPARK_PYTHON'] = sys.executable
# 	os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# 	return SparkSession \
# 		.builder \
# 		.appName(app_name) \
# 		.master("local[*]") \
# 		.config("spark.cores.max", max_cores) \
# 		.config("spark.executor.memory", "10g") \
# 		.config("spark.driver.memory", "10g") \
# 		.config("spark.memory.offHeap.enabled", True) \
# 		.config("spark.memory.offHeap.size", "10g") \
# 		.getOrCreate()