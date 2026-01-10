#!/bin/bash
# Tạo tiêu đề cho file CSV nếu file chưa tồn tại
if [ ! -f gpu_log.csv ]; then
    echo "timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB]" > gpu_log.csv
fi

while true
do
    # Lấy thông tin từ nvidia-smi và ghi vào file
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits >> gpu_log.csv
    
    # Chỉ giữ lại dòng tiêu đề (dòng 1) và 100 dòng cuối cùng
    # Tổng cộng file sẽ có tối đa 101 dòng
    echo "$(head -n 1 gpu_log.csv && tail -n 5 gpu_log.csv)" > gpu_log.csv
    
    sleep 2  # Nghỉ 2 giây trước khi lấy dữ liệu tiếp theo
done