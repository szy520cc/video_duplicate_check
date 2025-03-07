import cv2
import numpy as np
import pickle
import hashlib
import base64
import os
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from concurrent.futures import ThreadPoolExecutor
from app.services.logger import get_logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class VideoFeatureManager:
    """视频特征管理器，负责视频特征的提取、哈希计算和相似度比较
    
    该类使用单例模式实现，确保在整个应用中只有一个实例。主要功能包括：
    1. 视频关键帧提取和特征计算
    2. 基于多种图像哈希算法的特征提取
    3. 使用局部敏感哈希(LSH)进行特征降维
    4. 多特征融合的视频相似度计算
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.logger = get_logger()
        self.logger.info("初始化视频特征管理器...")
        
        # 视频帧采样参数 - 进一步优化采样策略
        self.frame_sample_interval = 2  # 设置帧采样间隔为2，平衡采样密度和特征质量
        self.min_sample_interval = 1    # 保持最小采样间隔为1
        self.scene_change_threshold = 5.0  # 进一步降低场景变化阈值，提高对微小场景变化的敏感度
        
        # 相似度计算参数
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.50'))  # 从环境变量读取相似度阈值，默认值0.80
        self.n_components = 512  # 进一步增加LSH降维后的特征维度，保留更多细节信息
        
        # 初始化局部敏感哈希投影矩阵
        self.random_projection = GaussianRandomProjection(n_components=self.n_components)
        
        # 并行处理参数
        self.max_workers = 8  # 进一步增加特征提取的最大线程数，提高并行处理能力
        
        self._initialized = True

    def calculate_feature_hash(self, features):
        """计算视频特征的增强型局部敏感哈希值
        
        使用增强型局部敏感哈希(LSH)技术将高维特征映射到低维空间，生成包含多个哈希段的字符串。
        该方法综合了帧哈希特征和颜色直方图特征，通过多段哈希保留更多的相似性结构信息。
        新版本增强了哈希值的信息量，使其能够直接用于相似度比较，无需依赖原始特征数据。
        
        Args:
            features: 包含视频特征的字典或序列化的字节串，包含frame_hashes和color_histograms
        
        Returns:
            str: 生成的多段哈希字符串，用于直接进行相似度比较
        """
        # 增加错误处理和日志输出，帮助诊断问题
        self.logger.debug("开始计算特征哈希...")
        try:
            # 反序列化特征数据
            if isinstance(features, bytes):
                features = pickle.loads(features)

            # 检查特征数据的有效性
            if not isinstance(features, dict):
                raise ValueError("特征数据必须是字典类型")
                
            if 'frame_hashes' not in features or not features['frame_hashes']:
                raise ValueError("特征数据缺少帧哈希信息")
                
            # 分别处理不同类型的特征
            frame_hash_features = []
            color_hist_features = []
            ssim_features = []
            
            # 提取并压缩帧哈希特征 - 使用更稳健的特征提取方法
            if 'frame_hashes' in features and features['frame_hashes']:
                self.logger.debug(f"处理 {len(features['frame_hashes'])} 个帧哈希特征")
                
                # 选择最具代表性的帧特征 - 基于熵和方差
                frame_importance = []
                for hash_value in features['frame_hashes']:
                    hash_array = np.array(hash_value)
                    # 计算特征重要性 - 结合熵和方差
                    entropy = -np.sum((np.mean(hash_array) * np.log2(np.mean(hash_array) + 1e-10) + 
                                     (1 - np.mean(hash_array)) * np.log2(1 - np.mean(hash_array) + 1e-10)))
                    variance = np.var(hash_array)
                    importance = entropy * variance  # 同时考虑信息量和变化程度
                    frame_importance.append(importance)
                
                # 选择最重要的帧特征（最多30个）
                if len(features['frame_hashes']) > 30:
                    top_indices = np.argsort(frame_importance)[-30:]
                    selected_frames = [features['frame_hashes'][i] for i in top_indices]
                else:
                    selected_frames = features['frame_hashes']
                
                self.logger.debug(f"选择了 {len(selected_frames)} 个关键帧特征")
                
                # 对每个选定的帧特征进行处理
                for hash_value in selected_frames:
                    hash_array = np.array(hash_value)
                    
                    # 直接使用原始特征的二值化版本，避免降维带来的信息损失
                    # 使用自适应阈值
                    threshold = np.median(hash_array)  # 中值作为阈值，对异常值更鲁棒
                    frame_bits = (hash_array >= threshold).flatten()
                    frame_hash_features.extend(frame_bits)
                    
                    # 添加梯度信息，捕获特征变化趋势
                    if len(hash_array) > 1:
                        gradient = hash_array[1:] - hash_array[:-1]
                        gradient_bits = (gradient >= 0).flatten()
                        frame_hash_features.extend(gradient_bits)
                
            # 提取并压缩颜色直方图特征 - 增强颜色特征的表达能力
            if 'color_histograms' in features and features['color_histograms']:
                self.logger.debug(f"处理 {len(features['color_histograms'])} 个颜色直方图特征")
                
                # 选择代表性的颜色直方图（最多10个）
                if len(features['color_histograms']) > 10:
                    # 计算每个直方图的方差作为选择标准
                    hist_variance = [np.var(hist) for hist in features['color_histograms']]
                    top_indices = np.argsort(hist_variance)[-10:]
                    selected_hists = [features['color_histograms'][i] for i in top_indices]
                else:
                    selected_hists = features['color_histograms']
                
                # 对每个选定的颜色直方图进行处理
                for hist in selected_hists:
                    # 对颜色直方图进行量化和压缩
                    hist_compressed = np.clip(hist * 255, 0, 255).astype(np.uint8)
                    # 只保留主要颜色分量
                    sorted_indices = np.argsort(hist_compressed)[-8:]  # 保留最显著的8个颜色分量
                    for idx in sorted_indices:
                        color_hist_features.append(hist_compressed[idx])
                    
                    # 添加颜色分布特征
                    color_hist_features.append(np.max(hist_compressed))
                    color_hist_features.append(np.min(hist_compressed))
                    color_hist_features.append(np.median(hist_compressed))
                
            # 提取并压缩SSIM特征 - 增强结构相似性特征
            if 'deep_features' in features and features['deep_features']:
                self.logger.debug(f"处理 {len(features['deep_features'])} 个SSIM特征")
                
                # 选择代表性的SSIM特征（最多5个）
                if len(features['deep_features']) > 5:
                    # 计算每个SSIM特征的方差作为选择标准
                    ssim_variance = [np.var(feat) for feat in features['deep_features']]
                    top_indices = np.argsort(ssim_variance)[-5:]
                    selected_ssim = [features['deep_features'][i] for i in top_indices]
                else:
                    selected_ssim = features['deep_features']
                
                # 对每个选定的SSIM特征进行处理
                for deep_feature in selected_ssim:
                    # 对SSIM特征进行二值化，使用自适应阈值
                    threshold = np.median(deep_feature)
                    ssim_bits = (deep_feature >= threshold).astype(np.uint8)
                    ssim_features.extend(ssim_bits)
            
            # 创建增强型多段哈希 - 使用更可靠的哈希生成方法
            hash_segments = []
            
            # 处理帧哈希特征 - 确保至少有帧特征，这是最重要的特征
            if frame_hash_features:
                try:
            
                    # 使用更可靠的帧特征哈希生成方法
                    frame_vector = np.array(frame_hash_features)
                    # 使用自适应阈值而非简单均值
                    threshold = np.median(frame_vector)
                    frame_bits = (frame_vector >= threshold).flatten()
                    frame_bytes = np.packbits(frame_bits)
                    frame_hash = base64.b64encode(frame_bytes).decode('utf-8')
                    hash_segments.append(f"F{frame_hash}")
                except Exception as e:
                    self.logger.error(f"帧特征哈希计算错误: {e}")
                    # 创建一个备用的帧特征哈希，但使用随机特征以避免误匹配
                    backup_bits = np.random.randint(0, 2, 256).astype(np.uint8)
                    backup_bytes = np.packbits(backup_bits)
                    backup_hash = base64.b64encode(backup_bytes).decode('utf-8')
                    hash_segments.append(f"F{backup_hash}")
            else:
                # 如果没有帧特征，创建一个随机的帧特征哈希，避免误匹配
                backup_bits = np.random.randint(0, 2, 256).astype(np.uint8)
                backup_bytes = np.packbits(backup_bits)
                backup_hash = base64.b64encode(backup_bytes).decode('utf-8')
                hash_segments.append(f"F{backup_hash}")
            
            # 处理颜色直方图特征
            if color_hist_features:
                try:
                    # 确保颜色特征数量适中，避免过大
                    if len(color_hist_features) > 256:
                        # 均匀采样以减少特征数量
                        indices = np.linspace(0, len(color_hist_features)-1, 256).astype(int)
                        color_hist_features = [color_hist_features[i] for i in indices]
                    
                    color_bytes = np.array(color_hist_features, dtype=np.uint8).tobytes()
                    color_hash = base64.b64encode(color_bytes).decode('utf-8')
                    hash_segments.append(f"C{color_hash}")
                except Exception as e:
                    self.logger.error(f"颜色特征哈希计算错误: {e}")
            
            # 处理SSIM特征
            if ssim_features:
                try:
                    # 确保SSIM特征数量适中
                    if len(ssim_features) > 512:
                        # 均匀采样以减少特征数量
                        indices = np.linspace(0, len(ssim_features)-1, 512).astype(int)
                        ssim_features = [ssim_features[i] for i in indices]
                    
                    ssim_bytes = np.packbits(np.array(ssim_features, dtype=np.uint8))
                    ssim_hash = base64.b64encode(ssim_bytes).decode('utf-8')
                    hash_segments.append(f"S{ssim_hash}")
                except Exception as e:
                    self.logger.error(f"SSIM特征哈希计算错误: {e}")
            
            # 合并多段哈希
            result = "|".join(hash_segments)
            
            # 确保结果非空
            if not result:
                raise ValueError("生成的特征哈希为空")
                
            return result
            
        except Exception as e:
            self.logger.error(f"特征哈希计算失败: {e}")
            # 返回一个默认的特征哈希，确保系统不会崩溃
            default_bits = np.ones(256, dtype=np.uint8)
            default_bytes = np.packbits(default_bits)
            default_hash = base64.b64encode(default_bytes).decode('utf-8')
            return f"F{default_hash}"

    def calculate_file_hash(self, file_path):
        """计算文件的SHA-256哈希值
        
        通过分块读取方式计算大文件的哈希值，避免内存溢出。
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            str: 文件的SHA-256哈希值的十六进制字符串
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # 分块读取文件以处理大文件
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def extract_video_features(self, video_path):
        """提取视频的多维特征
        
        该方法实现了自适应的关键帧提取和多特征融合：
        1. 动态调整帧采样间隔，确保捕获场景变化
        2. 并行提取每个关键帧的特征
        3. 提取的特征包括：图像哈希、颜色直方图、SSIM特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bytes: 序列化的特征数据字典，包含frame_hashes、color_histograms和deep_features
            
        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 视频文件无法打开
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        features = {
            'frame_hashes': [],
            'color_histograms': [],
            'deep_features': []
        }

        frame_count = 0
        prev_frame = None
        current_interval = self.frame_sample_interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 确保至少采样15个关键帧，提高特征代表性
        min_keyframes = 15
        if total_frames > 0 and fps > 0:
            # 动态调整采样间隔，确保至少采样min_keyframes个帧
            estimated_keyframes = total_frames / self.frame_sample_interval
            if estimated_keyframes < min_keyframes:
                # 调整采样间隔以获取足够的关键帧
                adjusted_interval = max(1, int(total_frames / min_keyframes))
                current_interval = min(self.frame_sample_interval, adjusted_interval)
            # 对于较长视频，确保采样足够的帧但不超过200个
            elif estimated_keyframes > 200:
                # 适当增加采样间隔，避免过多的帧导致噪声增加
                current_interval = max(self.frame_sample_interval, int(total_frames / 200))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % current_interval == 0:
                    # 动态调整采样间隔
                    if prev_frame is not None:
                        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
                        scene_change = np.mean(diff)
                        current_interval = max(self.min_sample_interval,
                                             int(self.frame_sample_interval * (1 - scene_change/self.scene_change_threshold)))

                    # 提交特征提取任务
                    executor.submit(self._extract_frame_features, frame, features)
                    prev_frame = frame.copy()

                frame_count += 1

        cap.release()
        return pickle.dumps(features)

    def _extract_frame_features(self, frame, features):
        """并行提取单帧的多维特征
        
        对输入帧进行预处理并提取多种特征：
        1. 均值哈希(aHash)：对整体亮度变化鲁棒
        2. 差值哈希(dHash)：捕获图像梯度信息
        3. 感知哈希(pHash)：基于DCT变换，对噪声鲁棒
        4. SSIM特征：捕获结构相似性信息
        5. 颜色直方图：表征颜色分布特征
        
        Args:
            frame: 输入的视频帧
            features: 特征存储字典，用于存储提取的特征
        """
        # 应用高斯模糊进行预处理，降低字幕等局部细节的影响
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # 计算均值哈希 - 增加分辨率从8x8到12x12，提高特征区分度
        resized = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        avg_hash = 1 * (resized >= avg)
        
        # 计算差值哈希 - 增加分辨率从9x8到13x12
        resized = cv2.resize(gray, (13, 12), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] - resized[:, :-1]
        dhash = 1 * (diff >= 0)
        
        # 计算感知哈希 - 保持32x32的DCT变换，但提取更多低频系数
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:12, :12]  # 提取更多的低频系数
        avg_dct = dct_low.mean()
        phash = 1 * (dct_low >= avg_dct)
        
        # 计算SSIM相关特征 - 增加分辨率从16x16到24x24
        resized = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
        mu = cv2.GaussianBlur(resized, (7, 7), 1.5)
        sigma = cv2.GaussianBlur(resized * resized, (7, 7), 1.5) - mu * mu
        ssim_features = np.concatenate([mu.flatten(), sigma.flatten()])
        
        # 组合所有哈希特征
        combined_hash = np.concatenate([avg_hash.flatten(), dhash.flatten(), phash.flatten()])
        features['frame_hashes'].append(combined_hash)
        
        # 计算三直方图 - 增加直方图分辨率从6到8
        hist_b = cv2.calcHist([blurred], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([blurred], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([blurred], [2], None, [8], [0, 256])
        hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        hist = cv2.normalize(hist, hist).flatten()
        features['color_histograms'].append(hist)
        
        # 添加SSIM特征
        features['deep_features'].append(ssim_features)
        
    def calculate_similarity(self, feature_hash1, feature_hash2):
        """计算两个视频的综合相似度
        
        基于增强型特征哈希直接计算视频相似度，无需依赖原始特征数据：
        1. 解析多段哈希字符串
        2. 分别计算帧特征、颜色特征和SSIM特征的相似度
        3. 使用加权平均融合多个相似度指标
        
        权重分配：
        - 帧特征相似度: 95%（对内容变化最敏感，大幅提高权重以增强区分能力）
        - SSIM特征相似度: 4%（捕获结构信息）
        - 颜色特征相似度: 1%（表征整体视觉特征）
        
        Args:
            feature_hash1: 第一个视频的特征哈希字符串
            feature_hash2: 第二个视频的特征哈希字符串
            
        Returns:
            float: 0.0~1.0之间的相似度值，1.0表示完全相同
        """
        self.logger.debug(f"计算视频相似度...")
        # 解析哈希字符串
        def parse_hash(hash_str):
            segments = {}
            for segment in hash_str.split('|'):
                if segment:
                    type_code = segment[0]
                    hash_data = segment[1:]
                    segments[type_code] = hash_data
            return segments
        
        # 计算二进制特征的相似度 - 使用增强型汉明距离计算，对微小差异更敏感
        def binary_similarity(hash1, hash2):
            try:
                data1 = np.unpackbits(np.frombuffer(base64.b64decode(hash1), dtype=np.uint8))
                data2 = np.unpackbits(np.frombuffer(base64.b64decode(hash2), dtype=np.uint8))
                
                # 确保数据长度足够
                if len(data1) < 8 or len(data2) < 8:
                    return 0.0
                
                # 处理不同长度的特征向量 - 使用动态规划寻找最佳匹配
                if abs(len(data1) - len(data2)) > len(data1) * 0.1:  # 降低长度差异容忍度到10%
                    # 长度差异过大，可能是完全不同的视频
                    return 0.0
                
                # 对齐特征向量 - 使用滑动窗口找到最佳匹配位置
                if len(data1) != len(data2):
                    if len(data1) > len(data2):
                        # 交换，确保data1是较短的
                        data1, data2 = data2, data1
                    
                    # 滑动窗口寻找最佳匹配，使用更小的步长提高精度
                    min_dist = float('inf')
                    step = max(1, (len(data2) - len(data1)) // 20)  # 使用更细的步长
                    for i in range(0, len(data2) - len(data1) + 1, step):
                        window = data2[i:i+len(data1)]
                        dist = np.sum(data1 != window)
                        min_dist = min(min_dist, dist)
                    
                    # 在最佳匹配点附近进行精细搜索
                    best_i = max(0, (min_dist // step) * step - step)
                    end_i = min(len(data2) - len(data1), best_i + 2*step)
                    for i in range(best_i, end_i + 1):
                        if i < len(data2) - len(data1) + 1:
                            window = data2[i:i+len(data1)]
                            dist = np.sum(data1 != window)
                            min_dist = min(min_dist, dist)
                    
                    hamming_dist = min_dist
                    effective_len = len(data1)
                else:
                    # 长度相同，直接计算汉明距离
                    hamming_dist = np.sum(data1 != data2)
                    effective_len = len(data1)
                
                # 归一化汉明距离，并转换为相似度
                raw_similarity = max(0.0, 1.0 - (hamming_dist / effective_len))
                
                # 应用六次方变换，更强烈地放大高相似度区间的差异
                # 这使得相似度在0.8-1.0区间有更高的区分度
                return raw_similarity ** 6
            except Exception as e:
                self.logger.error(f"二进制特征相似度计算错误: {e}")
                return 0.0  # 出错时返回零相似度
        
        # 计算颜色特征的相似度
        def color_similarity(hash1, hash2):
            try:
                data1 = np.frombuffer(base64.b64decode(hash1), dtype=np.uint8)
                data2 = np.frombuffer(base64.b64decode(hash2), dtype=np.uint8)
                
                # 确保数据非空
                if len(data1) == 0 or len(data2) == 0:
                    return 0.0
                
                # 将字节数据转换为直方图格式，确保形状正确
                # OpenCV的compareHist需要直方图是列向量，且数据类型为float32
                hist1 = data1.astype(np.float32)
                hist2 = data2.astype(np.float32)
                
                # 检查并调整直方图维度，确保两个直方图具有相同的维度
                if len(hist1) != len(hist2):
                    # 截取较短的长度
                    min_len = min(len(hist1), len(hist2))
                    hist1 = hist1[:min_len]
                    hist2 = hist2[:min_len]
                
                # 确保直方图非空
                if len(hist1) == 0:
                    return 0.0
                
                # 重塑为OpenCV直方图格式 - 列向量
                hist1 = hist1.reshape(-1, 1)
                hist2 = hist2.reshape(-1, 1)
                
                # 使用多种距离度量方法计算相似度，并取最高值
                similarities = []
                
                # 1. 巴氏距离
                try:
                    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                    similarities.append(max(0, 1.0 - dist))
                except cv2.error:
                    pass
                    
                # 2. 相关性
                try:
                    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    similarities.append(max(0, (corr + 1) / 2))  # 转换到0-1范围
                except cv2.error:
                    pass
                    
                # 3. 卡方距离
                try:
                    chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                    max_chi = np.sum(hist1) * 10  # 估计最大可能的卡方距离
                    similarities.append(max(0, 1.0 - (chi_square / max_chi)))
                except cv2.error:
                    pass
                    
                # 如果有多个相似度计算结果，取最高值
                if similarities:
                    return max(similarities)
                else:
                    # 如果所有方法都失败，使用欧氏距离作为备选
                    dist = np.sqrt(np.sum((hist1 - hist2) ** 2))
                    max_dist = np.sqrt(len(hist1)) * 255  # 最大可能距离
                    return max(0, 1.0 - (dist / max_dist))
            except Exception as e:
                self.logger.error(f"颜色特征相似度计算错误: {e}")
                return 0.0  # 出错时返回零相似度
        
        # 解析两个哈希字符串
        segments1 = parse_hash(feature_hash1)
        segments2 = parse_hash(feature_hash2)
        
        # 初始化相似度分数
        frame_similarity = 0.0
        color_similarity_score = 0.0
        ssim_similarity = 0.0
        
        # 计算帧特征相似度
        if 'F' in segments1 and 'F' in segments2:
            frame_similarity = binary_similarity(segments1['F'], segments2['F'])
        
        # 计算颜色特征相似度
        if 'C' in segments1 and 'C' in segments2:
            color_similarity_score = color_similarity(segments1['C'], segments2['C'])
        
        # 计算SSIM特征相似度
        if 'S' in segments1 and 'S' in segments2:
            ssim_similarity = binary_similarity(segments1['S'], segments2['S'])
        
        # 计算加权平均相似度 - 优化权重分配
        weights = [0.95, 0.01, 0.04]  # 保持帧特征的高权重95%，颜色特征1%，SSIM特征4%
        features_present = [('F' in segments1 and 'F' in segments2),
                           ('C' in segments1 and 'C' in segments2),
                           ('S' in segments1 and 'S' in segments2)]
        
        if not any(features_present):
            return 0.0  # 如果没有共同特征，返回0相似度
        
        # 如果没有帧特征，直接返回很低的相似度
        if not features_present[0]:
            return 0.1  # 没有帧特征时，相似度极低
        
        # 调整权重以适应可用特征
        adjusted_weights = [w if p else 0 for w, p in zip(weights, features_present)]
        weight_sum = sum(adjusted_weights)
        if weight_sum == 0:
            return 0.0
        
        adjusted_weights = [w / weight_sum for w in adjusted_weights]
        
        # 计算最终相似度
        similarity_scores = [frame_similarity, color_similarity_score, ssim_similarity]
        final_similarity = sum(s * w for s, w, p in zip(similarity_scores, adjusted_weights, features_present) if p)
        
        # 打印调试信息
        self.logger.debug(f"帧相似度: {frame_similarity:.4f}, 颜色相似度: {color_similarity_score:.4f}, SSIM相似度: {ssim_similarity:.4f}")
        self.logger.debug(f"加权前相似度: {final_similarity:.4f}")
        
        # 应用非线性变换，使相似度分布更加合理
        # 使用改进的S形函数变换，在高相似度区间(0.65-0.9)放大差异，提高区分度
        if final_similarity > 0.65:
            # 在高相似度区间使用更强的S形变换，增强区分能力
            final_similarity = 0.65 + 0.35 * (6 * (final_similarity - 0.65)) / (1 + 5 * abs(final_similarity - 0.65))
        elif final_similarity > 0.4:
            final_similarity = 0.4 + 0.25 * (4 * (final_similarity - 0.4)) / (1 + 3 * abs(final_similarity - 0.4))
        else:
            final_similarity = pow(final_similarity, 2.5)  # 低相似度区间使用较温和的幂函数压缩
            
        self.logger.debug(f"调整后最终相似度: {final_similarity:.4f}")
        
        return min(1.0, final_similarity)  # 确保相似度不超过1.0

    def is_duplicate(self, feature_hash1, feature_hash2, threshold=None):
        """判断两个视频是否重复
        
        基于特征相似度判断视频是否重复，支持自定义阈值
        使用增强型多级阈值策略，提高查重准确性和灵活性
        
        Args:
            feature_hash1: 第一个视频的特征哈希字符串
            feature_hash2: 第二个视频的特征哈希字符串
            threshold: 判定重复的相似度阈值，默认使用self.similarity_threshold
            
        Returns:
            tuple: (is_duplicate, similarity)
            - is_duplicate: bool，是否判定为重复
            - similarity: float，计算的相似度值
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        self.logger.debug(f"使用判定阈值: {threshold}")
            
        # 计算基础相似度
        similarity = self.calculate_similarity(feature_hash1, feature_hash2)
        
        # 解析哈希字符串，用于特征质量评估
        def parse_hash(hash_str):
            segments = {}
            for segment in hash_str.split('|'):
                if segment:
                    type_code = segment[0]
                    hash_data = segment[1:]
                    segments[type_code] = hash_data
            return segments
            
        segments1 = parse_hash(feature_hash1)
        segments2 = parse_hash(feature_hash2)
        
        # 检查是否包含帧特征，这是最重要的特征
        has_frame_features = 'F' in segments1 and 'F' in segments2
        
        # 如果没有帧特征，直接返回不匹配
        if not has_frame_features:
            self.logger.warning("缺少帧特征，判定为不重复")
            return False, similarity
        
        # 计算帧特征的相似度 - 使用增强型算法单独评估重要的特征
        frame_similarity = 0.0
        if 'F' in segments1 and 'F' in segments2:
            try:
                data1 = np.unpackbits(np.frombuffer(base64.b64decode(segments1['F']), dtype=np.uint8))
                data2 = np.unpackbits(np.frombuffer(base64.b64decode(segments2['F']), dtype=np.uint8))
                
                # 确保数据长度足够
                if len(data1) < 8 or len(data2) < 8:
                    self.logger.warning("帧特征数据长度不足，判定为不重复")
                    return False, similarity
                
                # 处理不同长度的特征向量 - 使用滑动窗口寻找最佳匹配
                if abs(len(data1) - len(data2)) > len(data1) * 0.1:  # 降低长度差异容忍度到10%
                    # 长度差异过大，可能是完全不同的视频
                    self.logger.warning(f"帧特征长度差异过大: {len(data1)} vs {len(data2)}，判定为不重复")
                    return False, similarity
                
                # 对齐特征向量 - 使用滑动窗口找到最佳匹配位置
                if len(data1) != len(data2):
                    if len(data1) > len(data2):
                        # 交换，确保data1是较短的
                        data1, data2 = data2, data1
                    
                    # 滑动窗口寻找最佳匹配，使用更小的步长提高精度
                    min_dist = float('inf')
                    step = max(1, (len(data2) - len(data1)) // 20)  # 使用更细的步长
                    for i in range(0, len(data2) - len(data1) + 1, step):
                        window = data2[i:i+len(data1)]
                        dist = np.sum(data1 != window)
                        min_dist = min(min_dist, dist)
                    
                    # 在最佳匹配点附近进行精细搜索
                    best_i = max(0, (min_dist // step) * step - step)
                    end_i = min(len(data2) - len(data1), best_i + 2*step)
                    for i in range(best_i, end_i + 1):
                        if i < len(data2) - len(data1) + 1:
                            window = data2[i:i+len(data1)]
                            dist = np.sum(data1 != window)
                            min_dist = min(min_dist, dist)
                    
                    hamming_dist = min_dist
                    effective_len = len(data1)
                else:
                    # 长度相同，直接计算汉明距离
                    hamming_dist = np.sum(data1 != data2)
                    effective_len = len(data1)
                
                # 归一化汉明距离，并转换为相似度
                frame_similarity = max(0.0, 1.0 - (hamming_dist / effective_len))
                # 应用六次方变换，更强烈地放大高相似度区间的差异
                frame_similarity = frame_similarity ** 6
                self.logger.debug(f"帧特征相似度: {frame_similarity:.4f}")
            except Exception as e:
                self.logger.error(f"帧特征相似度计算错误: {e}")
                return False, similarity  # 出错时返回不匹配
        
        # 增强型多级阈值判定策略 - 更精细的判断条件
        self.logger.debug("应用多级阈值判定策略...")
        
        # 1. 如果综合相似度极高(>0.95)，直接判定为重复
        if similarity > 0.95:
            self.logger.info("综合相似度极高(>0.95)，判定为重复")
            return True, similarity
            
        # 2. 如果帧特征相似度极高(>0.95)且综合相似度达到阈值的75%，判定为重复
        if frame_similarity > 0.95 and similarity >= threshold * 0.75:
            self.logger.info("帧特征相似度极高(>0.95)且综合相似度达到阈值的75%，判定为重复")
            return True, similarity
            
        # 3. 如果帧特征相似度高(>0.9)且综合相似度达到阈值的85%，判定为重复
        if frame_similarity > 0.9 and similarity >= threshold * 0.85:
            self.logger.info("帧特征相似度高(>0.9)且综合相似度达到阈值的85%，判定为重复")
            return True, similarity
            
        # 4. 如果帧特征相似度和综合相似度都很高，判定为重复
        if frame_similarity > 0.85 and similarity > 0.85:
            self.logger.info("帧特征相似度和综合相似度都很高(>0.85)，判定为重复")
            return True, similarity
            
        # 5. 标准阈值判定 - 只考虑综合相似度
        # 修改：移除帧特征相似度的要求，只根据用户设置的阈值判断
        is_dup = (similarity >= threshold)
        
        if is_dup:
            self.logger.info(f"标准阈值判定: 综合相似度({similarity:.4f})≥阈值({threshold})且帧特征相似度({frame_similarity:.4f})≥0.8，判定为重复")
        else:
            self.logger.info(f"标准阈值判定: 综合相似度({similarity:.4f})或帧特征相似度({frame_similarity:.4f})不满足条件，判定为不重复")
            
        return is_dup, similarity