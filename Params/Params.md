## 视频收集命令

```shell
csi-record \
  --camera 0 \
  --width 640 \
  --height 360 \
  --fps 30 \
  --duration 0 \
  --sample-id S03 \
  --out-dir /home/lyc/Desktop/Video/S00 \
  --show
```

## TX 发送命令

```shell
sudo feitcsi   --mode inject   --frequency 5180   --channel-width 80   --format HESU   --mcs 0   --spatial-streams 1   --antenna 1   --tx-power 10   --inject-delay 10000   -v
```



## RX 接收脚本

["D:\Desktop\AXHmm\Params\collect_csi_formal.sh"]



## 剩余参数

所有房间的摄像头离地高度均为 1.8 m

所有 Tx、Rx 离地高度均为 0.70 m

客厅 Tx–Rx 距离 3.0 m

卧室 Tx–Rx 距离 3.1 m

卫生间 Tx–Rx 距离 2.1 m

厨房 Tx–Rx 距离 3.0 m

客厅大小：6.9 m × 3.6 m

卧室大小：3.05 m × 3.3 m

厨房大小：4.5 m × 1.6 m

卫生间大小：2.25 m × 2.1 m
