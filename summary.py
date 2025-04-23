import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, col, lit, coalesce, count, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# 初始化Spark
spark = SparkSession.builder.appName("DrivingBehavior").getOrCreate()

# 获取当前路径并构造data文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "*")

# 定义19列的schema
columns = [
    "driverID", "carPlateNumber", "Latitude", "Longitude", "Speed",
    "Direction", "siteName", "Time", "isRapidlySpeedup", "isRapidlySlowdown",
    "isNeutralSlide", "isNeutralSlideFinished", "neutralSlideTime",
    "isOverspeed", "isOverspeedFinished", "overspeedTime", "isFatigueDriving",
    "isHthrottleStop", "isOilLeak"
]
schema = StructType([StructField(col, StringType(), True) for col in columns])

# 读取所有文件，强制19列
df = spark.read.option("delimiter", ",") \
               .option("mode", "PERMISSIVE") \
               .csv(data_path, header=False, schema=schema)

# 应用列名
df = df.toDF(*columns)

# 强制类型转换并处理空值
df = df.withColumn("isOverspeed", coalesce(col("isOverspeed").cast(IntegerType()), lit(0))) \
       .withColumn("isOverspeedFinished", coalesce(col("isOverspeedFinished").cast(IntegerType()), lit(0))) \
       .withColumn("isFatigueDriving", coalesce(col("isFatigueDriving").cast(IntegerType()), lit(0))) \
       .withColumn("isNeutralSlide", coalesce(col("isNeutralSlide").cast(IntegerType()), lit(0))) \
       .withColumn("isNeutralSlideFinished", coalesce(col("isNeutralSlideFinished").cast(IntegerType()), lit(0))) \
       .withColumn("overspeedTime", coalesce(col("overspeedTime").cast(DoubleType()), lit(0.0))) \
       .withColumn("neutralSlideTime", coalesce(col("neutralSlideTime").cast(DoubleType()), lit(0.0)))

# 调试：检查原始数据
print("Raw data sample:")
df.select("carPlateNumber", "isOverspeed", "isOverspeedFinished", "overspeedTime", "isFatigueDriving", "isNeutralSlide", "isNeutralSlideFinished", "neutralSlideTime").show(20, truncate=False)

# 按车牌号统计
summary = df.groupBy("carPlateNumber").agg(
    # 超速次数：统计isOverspeedFinished=1的记录数（超速事件结束）
    count(when(col("isOverspeedFinished") == 1, True)).alias("overspeed_count"),
    # 疲劳驾驶次数：统计isFatigueDriving=1的记录数
    count(when(col("isFatigueDriving") == 1, True)).alias("fatigue_count"),
    # 空挡滑行次数：统计isNeutralSlideFinished=1的记录数（空挡滑行事件结束）
    count(when(col("isNeutralSlideFinished") == 1, True)).alias("neutral_slide_count"),
    # 超速总时间：仅累加isOverspeedFinished=1时的overspeedTime
    sum(when(col("isOverspeedFinished") == 1, col("overspeedTime")).otherwise(0)).alias("total_overspeed_time"),
    # 空挡滑行总时间：仅累加isNeutralSlideFinished=1时的neutralSlideTime
    sum(when(col("isNeutralSlideFinished") == 1, col("neutralSlideTime")).otherwise(0)).alias("total_neutral_slide_time")
)

# 调试：显示统计结果
print("Summary results:")
summary.show()

# 保存结果为单个JSON文件
output_path = os.path.join(current_dir, "output", "summary.json")
summary.coalesce(1).write.mode("overwrite").json(output_path)

# 关闭Spark
spark.stop()