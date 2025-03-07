# 视频查重系统

一个基于Python开发的高效视频查重系统，能够检测完全重复和相似的视频文件。系统使用文件哈希进行快速重复检测，并结合多维视频特征分析实现相似度比较。通过Redis缓存机制和并行处理提升查询性能。

## 主要特点

- **多维特征分析**
  - 均值哈希(aHash)：对亮度变化鲁棒
  - 差值哈希(dHash)：捕获图像梯度信息
  - 感知哈希(pHash)：基于DCT变换，抗噪声
  - SSIM结构相似度：分析图像结构信息
  - 颜色直方图：表征颜色分布特征
- **智能采样机制**
  - 自适应帧采样：根据场景变化动态调整
  - 并行特征提取：多线程处理提升效率
  - 关键帧分析：捕获视频关键信息
- **高效缓存**
  - 使用Redis缓存视频特征和哈希值
  - 支持缓存预热，提升系统启动性能
  - 特征值索引加速查询
- **批量处理**
  - 支持文件夹批量导入
  - 分批处理大量视频数据
  - 内存优化的大文件处理
- **可配置的相似度判定**
  - 多特征加权融合
  - 灵活的阈值设置
  - 默认阈值0.85（可调整）

## 环境要求

- Python 3.12+
- MySQL 数据库
- Redis 服务器
- 操作系统：支持 Windows/Linux/MacOS

## 依赖包

```
opencv-python>=4.5.0
numpy>=1.19.0
mysql-connector-python>=8.0.0
python-dotenv>=0.19.0
redis>=5.0.0
scikit-learn>=1.3.0
```

## 安装步骤

1. 克隆项目到本地

2. 创建并激活虚拟环境（推荐）
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   .\venv\Scripts\activate  # Windows
   ```

3. 安装依赖包
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量
   创建 `.env` 文件并设置以下参数：
   ```
   # MySQL数据库配置
   DB_HOST=localhost      # 数据库主机地址
   DB_PORT=3306          # 数据库端口
   DB_USER=your_username # 数据库用户名
   DB_PASSWORD=your_password # 数据库密码
   DB_NAME=your_database # 数据库名称

   # Redis配置
   REDIS_HOST=localhost   # Redis主机地址
   REDIS_PORT=6379       # Redis端口
   REDIS_DB=0            # Redis数据库编号
   REDIS_PASSWORD=       # Redis密码（如果有）

   # 系统配置
   CACHE_EXPIRE=3600     # 缓存过期时间（秒）
   BATCH_SIZE=100        # 批处理大小
   
   # 性能参数
   FRAME_SAMPLE_INTERVAL=8  # 基础帧采样间隔
   MIN_SAMPLE_INTERVAL=6    # 最小采样间隔
   SCENE_CHANGE_THRESHOLD=40.0  # 场景变化阈值
   MAX_WORKERS=4            # 特征提取线程数
   ```

   > 注意：
   > - 所有配置项都需要根据实际环境进行修改
   > - 建议将 `.env` 文件添加到 `.gitignore` 中以保护敏感信息
   > - 可以参考 `.env.development` 文件进行配置

5. 创建视频目录
   ```bash
   mkdir video
   ```

## 使用方法

1. 将待查重的视频文件放入 `video` 目录

2. 运行程序
   ```bash
   python main.py [--threshold 0.85] [--batch-size 100]
   ```
   参数说明：
   - `--threshold`：视频相似度判定阈值（0.0-1.0），默认0.85
   - `--batch-size`：批处理大小，默认100

3. 查看结果
   - 程序会显示重复视频的信息，包括文件路径和相似度
   - 非重复视频会被自动添加到数据库中

## 工作流程

1. **系统初始化**
   - 连接数据库和Redis
   - 创建必要的数据表
   - 执行缓存预热
   - 初始化特征提取器

2. **视频处理**
   - 计算文件SHA-256哈希值
   - 检查完全重复
   - 自适应关键帧提取
   - 并行特征计算
   - 局部敏感哈希降维
   - 计算综合相似度

3. **结果处理**
   - 显示重复视频信息
   - 添加新视频到数据库
   - 更新Redis缓存
   - 生成查重报告

## 错误处理

- **文件错误**
  - 检查视频文件完整性
  - 处理不支持的视频格式
  - 跳过损坏的视频文件

- **资源错误**
  - 自动重连数据库
  - Redis连接失败回退
  - 内存不足时分批处理

- **运行时错误**
  - 特征提取失败重试
  - 异常状态恢复
  - 错误日志记录

## 性能优化

- **特征提取优化**
  - 自适应帧采样减少计算量
  - 多线程并行处理提升速度
  - 预处理过滤无效帧

- **存储优化**
  - 批量数据库操作
  - Redis特征值缓存
  - 二级缓存机制

- **查询优化**
  - 特征值索引
  - 局部敏感哈希加速搜索
  - 多级过滤策略

## 技术栈

- Python：核心开发语言
- OpenCV：视频处理和特征提取
- NumPy：高性能数值计算
- scikit-learn：特征降维和机器学习
- MySQL：持久化存储
- Redis：高速缓存
- python-dotenv：环境配置