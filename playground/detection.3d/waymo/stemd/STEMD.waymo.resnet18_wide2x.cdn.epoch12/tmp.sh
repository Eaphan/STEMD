#!/bin/bash
#SBATCH -J walk                   # 作业名为 test
#SBATCH -o test.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -A pa_cs_department       #
#SBATCH -p special_cs             # 作业提交的分区为 cpu
##SBATCH --qos=debug               # 作业使用的 QoS 为 debug
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=32         # 单任务使用的 CPU 核心数为 4
##SBATCH -t 10-00:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100:4         # 单个节点使用 1 块 GPU 卡
##SBATCh -w hpc-gpu014                # 指定运行作业的节点是 comput6，若不填写系统自动分配节点

efg_run --num-gpus 4 &> log.txt
