# zhongguan-xinyuan-vr-sentiment
文本挖掘、机器学习（系统可视）

### 系统核心功能

| 模块 | 技术实现 | 作用 |
| :--- | :--- | :--- |
| **全景可视化** | Pannellum.js | 360° 房间漫游，提供沉浸式体验。 |
| **物体定位** | **YOLOv8x** | 自动识别房间内家具（如 chair, bed, couch）的位置坐标。 |
| **坐标转换** | Python (OpenCV / 数学函数) | 将 YOLO 识别出的 2D 像素坐标精确转换为 3D 全景图所需的 Pitch/Yaw 角度。 |
| **情感分析** | Rule-Based / **LLM API** (备选) | 解析评论文本，判断评论对象（Target: bed）和情感（Sentiment: 积极/消极）。 |
| **数据整合** | Flask (Python) / In-Memory DB | 实现评论提交、存储、查询，并将 AI 分析结果实时映射到 `data.json` 中。 |

---

### 环境搭建与运行

#### 1. 前提条件

请确保您的系统已安装 **Python 3.8+** 和以下依赖库。

```bash
pip install Flask ultralytics opencv-python requests

#### 3. 运行步骤 (Execution Steps)

**步骤一：生成初始化数据 (运行 YOLO 坐标提取)**

# 在项目根目录下执行（假设 test_yolo.py 放在根目录）
python test_yolo.py 

# (注意：执行此命令前，请确保 yolov8x.pt 文件已存在于根目录或 models/ 文件夹中。)

**步骤二：启动 Flask 服务器**

python app.py
