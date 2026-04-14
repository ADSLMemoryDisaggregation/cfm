sudo python scheduler.py 63 10.10.1.2:50051,10.10.1.3:50051,10.10.1.4:50051,10.10.1.5:50051,10.10.1.6:50051 \
32 81920 -r --size 200 --max_far 163840 \
--workload xsbench,pagerank,redis,snappy,xgboost --ratios 27:14:28:18:13 --random \
--until 200
