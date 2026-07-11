# LYKON Motion Lab

> Basketball Smart Wearable · IMU + UWB + AI Motion Intelligence

LYKON Motion Lab 是 LYKON 篮球智能护臂项目的核心研发仓库。

我们正在构建一套面向大众篮球的智能穿戴系统，通过双护臂 IMU、UWB 定位、视频蒸馏学习和 AI 事件推理技术，将真实比赛转化为可记录、可分析、可回放、可成长的数字篮球资产。

最终目标：

**让每一个普通篮球爱好者都拥有自己的数字球员。**

---

## Vision

职业球员拥有：

- 比赛录像
- 运动追踪系统
- 数据分析团队
- 球员成长档案

而大众篮球长期存在：

> 打完即消失

没有比赛数据

没有动作分析

没有成长记录

没有数字身份

LYKON 希望通过智能穿戴与运动 AI，让每一个普通球员都能拥有属于自己的：

- 比赛数据
- AI 分析报告
- 3D 比赛回放
- 数字球员成长档案

---

# What We Are Building

LYKON 不只是一个护臂。

我们正在构建：

```text
真实比赛
    ↓
动作感知
    ↓
事件理解
    ↓
比赛重建
    ↓
数字球员
```

核心产品路线：

### Phase 1

智能护臂 + 数据分析

### Phase 2

3D 比赛回放

### Phase 3

数字球员生态

---

# System Overview

## Training Stage

```text
IMU + UWB + Video
        ↓
Pose Extraction
        ↓
Manual Labels
        ↓
Teacher Dataset
        ↓
Motion Model Training
        ↓
Event Recognition
        ↓
3D Reconstruction
```

## Product Stage

```text
Dual-arm IMU + UWB
        ↓
Motion Recognition
        ↓
Basketball Event Engine
        ↓
Match Timeline
        ↓
3D Replay
        ↓
Digital Athlete
```

视频骨骼仅用于训练阶段。

最终产品目标是不依赖复杂场边摄像机，仅通过智能护臂和定位设备完成比赛感知与数字重建。

---

# Repository Structure

```text
LYKON-Motion-Lab/

├── data/
│
├── docs/
│
├── scripts/
│
├── models/
│
├── hardware/
│
├── notebooks/
│
├── H5/
│
└── outputs/
```

---

# Modules

## 1. Smart Wearable System

智能护臂负责采集：

- IMU 加速度
- IMU 陀螺仪
- 左右护臂同步数据
- UWB 空间定位数据

对应目录：

```text
hardware/
data/raw/imu/
data/raw/uwb/
```

---

## 2. Motion Recognition

负责识别篮球动作：

- idle
- walk_run
- sprint_drive
- jump
- dribble
- shot

后续扩展：

- pass
- layup
- rebound
- catch
- defense_slide

技术路线：

- Rule Engine
- XGBoost
- tsai
- Time Series Learning

---

## 3. Video Teacher Dataset

视频骨骼用于构建训练真值：

```text
video_metadata.csv
video_frames.csv
pose_keypoints_2d.csv
joint_angles_2d.csv
action_labels.csv
event_labels.csv
```

训练阶段：

```text
IMU + UWB + Video + Pose + Labels
```

产品阶段：

```text
IMU + UWB
```

---

## 4. Skeleton Reconstruction

骨骼轨迹重建模块负责：

- 上肢动作恢复
- 下肢动作补全
- 3D 姿态估计
- 数字人物驱动

当前已经完成：

```text
真实动作
    ↓
骨骼轨迹
    ↓
数字人物
```

核心链路验证。

---

## 5. Basketball Event Engine

动作识别只是第一步。

LYKON 更重要的是理解比赛。

例如：

```text
catch
 ↓
dribble
 ↓
drive
 ↓
jump
 ↓
shot
```

被系统识别为：

```text
突破上篮
```

最终生成：

- 比赛时间线
- 热力图
- 回合分析
- 球员报告
- 精彩回放

---

## 6. Digital Athlete

数字球员是项目最终形态。

每位用户拥有：

- 个人档案
- 比赛历史
- 能力值
- 风格标签
- 成长曲线
- 3D Avatar

未来支持：

- 球员对比
- 相似度分析
- 排行榜
- 虚拟赛事
- 游戏化玩法

---

# Frontend Demo

前端位于：

```text
H5/
```

当前包含：

- Player App
- 数字球员展示
- 数据分析页面
- 比赛复盘页面
- NBA2K 风格回放入口

本地运行：

```bash
cd H5
python3 -m http.server 4173
```

访问：

```text
http://127.0.0.1:4173/
```

---

# Quick Start

创建环境：

```bash
conda create -n lykon python=3.10 -y
conda activate lykon
```

安装依赖：

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python scripts/00_check_env.py
```

创建数据表：

```bash
python scripts/05_create_empty_tables.py
```

生成测试数据：

```bash
python scripts/01_generate_dummy_data.py
```

特征工程：

```bash
python scripts/02_build_features.py
```

训练模型：

```bash
python scripts/03_train_baseline.py
```

运行规则引擎：

```bash
python scripts/04_run_rules.py
```

---

# Data Principles

核心原则：

1. Raw 数据永不修改
2. Label 与 Raw 分离
3. 所有数据拥有 session_id
4. 微秒级时间戳统一对齐
5. 视频作为 Teacher，不是产品依赖
6. 所有模型结果可追溯
7. 大体积视频与模型不直接上传仓库

---

# Branch Guide

当前仓库历史上存在多个实验分支：

```text
main
demo
framework-public
imu
train
gh-pages
```

含义：

### main

主分支

完整项目入口

---

### imu

传感器采集与硬件实验

---

### train

动作识别与模型训练

---

### framework-public

公开版算法框架与文档

---

### demo

数字球员前端与产品 Demo

---

### gh-pages

GitHub Pages 部署分支

---

未来将逐步整合至：

```text
main
gh-pages
```

开发功能统一采用：

```text
feature/imu-pipeline
feature/motion-recognition
feature/skeleton-reconstruction
feature/frontend
```

 not building a smart arm sleeve.

> We are building a digital basketball identity for every player.
