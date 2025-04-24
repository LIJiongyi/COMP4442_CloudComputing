'''问题
速度图例有问题
警告框位置不应该在最下面
3秒更新一次
蓝色线更新有延迟
选择输入时间没有纠错
'''

from flask import Flask, render_template, jsonify, request
import json
import os
import glob
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, count, when, sum, min, max
from datetime import datetime, timedelta

app = Flask(__name__)

# 初始化Spark
spark = SparkSession.builder \
    .appName("SpeedMonitor") \
    .master("local[*]") \
    .config("spark.driver.memory", "1g") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# 定义schema
schema = StructType([StructField(col, StringType(), True) for col in [
    "driverID", "carPlateNumber", "Latitude", "Longitude", "Speed",
    "Direction", "siteName", "Time", "isRapidlySpeedup", "isRapidlySlowdown",
    "isNeutralSlide", "isNeutralSlideFinished", "neutralSlideTime",
    "isOverspeed", "isOverspeedFinished", "overspeedTime", "isFatigueDriving",
    "isHthrottleStop", "isOilLeak"
]])

# 加载所有数据
df = spark.read.option("delimiter", ",").option("mode", "PERMISSIVE").csv("data/*", header=False, schema=schema)
df = df.withColumn("Time", col("Time").cast("timestamp"))

# 查询数据集的时间范围
time_range = df.agg(min("Time").alias("min_time"), max("Time").alias("max_time")).collect()[0]
default_start_time = time_range["min_time"].strftime("%Y-%m-%d %H:%M:%S")
default_end_time = time_range["max_time"].strftime("%Y-%m-%d %H:%M:%S")
print(f"Dataset time range - Start: {default_start_time}, End: {default_end_time}")

# 缓存速度数据
speed_data = df.select("carPlateNumber", "Speed", "isOverspeed", "Time").orderBy("Time").collect()
car_plates = sorted(list(set(row["carPlateNumber"] for row in speed_data)))

# 加载summary.json并按车牌号排序
summary_data = []
json_path = os.path.join("output", "summary.json", "part-00000-*.json")
for file in glob.glob(json_path):
    with open(file, "r") as f:
        for line in f:
            summary_data.append(json.loads(line))
# 按carPlateNumber排序
summary_data = sorted(summary_data, key=lambda x: x["carPlateNumber"])
print("Initial summary_data (sorted):", summary_data)

# 首页：显示动态图表和静态表格
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        print(f"POST request - Start Time: {start_time}, End Time: {end_time}")
        if start_time and end_time:
            # 按时间段过滤数据
            filtered_df = df.filter((col("Time") >= start_time) & (col("Time") <= end_time))
            summary = filtered_df.groupBy("carPlateNumber").agg(
                count(when(col("isOverspeedFinished") == 1, True)).alias("overspeed_count"),
                count(when(col("isFatigueDriving") == 1, True)).alias("fatigue_count"),
                count(when(col("isNeutralSlideFinished") == 1, True)).alias("neutral_slide_count"),
                sum(when(col("isOverspeedFinished") == 1, col("overspeedTime")).otherwise(0)).alias("total_overspeed_time"),
                sum(when(col("isNeutralSlideFinished") == 1, col("neutralSlideTime")).otherwise(0)).alias("total_neutral_slide_time")
            ).orderBy("carPlateNumber").collect()  # 按carPlateNumber排序
            summary_data_filtered = [row.asDict() for row in summary]
            print("POST request - Filtered Summary (sorted):", summary_data_filtered)
            return render_template("index.html", summary=summary_data_filtered, start_time=start_time, end_time=end_time, car_plates=car_plates, chart_start_time=start_time)
    print("GET request - Default Summary (sorted):", summary_data)
    return render_template("index.html", summary=summary_data, start_time=default_start_time, end_time=default_end_time, car_plates=car_plates, chart_start_time=default_start_time)

# 速度数据API：根据时间跨度返回数据
# 使用字典存储每个车牌的当前索引，独立递增
index_dict = {car_plate: 0 for car_plate in car_plates}
@app.route("/speed/<car_plate>")
def get_speed(car_plate):
    start_time = request.args.get("start_time")
    # 找到该车牌的数据
    filtered_data = [row for row in speed_data if row["carPlateNumber"] == car_plate]
    if not filtered_data:
        return jsonify({"error": "Car not found"})
    
    # 每次请求都根据start_time重新设置索引
    if start_time:
        try:
            # 解码URL中的空格（%20）
            start_time = start_time.replace("%20", " ")
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            for i, row in enumerate(filtered_data):
                row_dt = row["Time"]
                if row_dt >= start_dt:
                    index_dict[car_plate] = i
                    break
        except ValueError as e:
            print(f"Error parsing start_time: {e}")
            return jsonify({"error": "Invalid start_time format"})
    else:
        start_time = default_start_time

    # 根据记录时间计算30秒内的实际条数
    batch_data = []
    current_index = index_dict[car_plate]
    if current_index >= len(filtered_data):
        current_index = 0  # 重置索引，避免越界
        index_dict[car_plate] = 0

    start_row = filtered_data[current_index]
    start_dt = start_row["Time"]
    end_dt = start_dt + timedelta(seconds=30)  # 30秒时间跨度

    # 从当前索引开始，获取30秒内的记录
    i = 0
    while current_index + i < len(filtered_data):
        row = filtered_data[current_index + i]
        row_dt = row["Time"]
        if row_dt <= end_dt:
            batch_data.append({
                "carPlateNumber": row["carPlateNumber"],
                "speed": float(row["Speed"]) if row["Speed"] else 0.0,
                "isOverspeed": int(row["isOverspeed"]) if row["isOverspeed"] else 0,
                "time": row["Time"].strftime("%Y-%m-%d %H:%M:%S")
            })
            i += 1
        else:
            break

    # 如果没有数据，返回空列表
    if not batch_data:
        print(f"No new data for {car_plate} within 30 seconds from {start_dt}")
        return jsonify([])

    # 更新索引
    index_dict[car_plate] = current_index + i
    print(f"Returning batch for {car_plate}, start index: {current_index}, batch size: {i}, batch data: {batch_data}")
    return jsonify(batch_data)

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', debug=True, port=8888)
    finally:
        spark.stop()

