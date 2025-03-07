import os
import hashlib
import requests
import shutil
import time
import re
from urllib.parse import urlparse
from fastapi import HTTPException
from app.services.logger import get_logger
from pathlib import Path
from typing import Optional

# 配置参数
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DOWNLOAD_TIMEOUT = 30  # 30秒

# 确保download目录存在
download_dir = Path("download")
download_dir.mkdir(exist_ok=True)

# 支持的视频文件扩展名
VALID_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
    '.webm', '.m4v', '.mpeg', '.mpg', '.3gp'
}


def check_file_size(file_path: str) -> None:
    """检查文件大小是否超过限制
    
    Args:
        file_path (str): 文件路径
        
    Raises:
        HTTPException: 当文件大小超过限制时抛出413错误
    """
    file_size_limit = int(os.getenv('FILE_SIZE_LIMIT', '50')) * 1024 * 1024  # 默认50MB
    if file_size_limit > 0:
        file_size = os.path.getsize(file_path)
        if file_size > file_size_limit:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制：{file_size_limit / (1024 * 1024):.1f}MB"
            )

def get_file_extension(url: str) -> str:
    """从URL中提取文件扩展名
    
    Args:
        url (str): 文件URL
        
    Returns:
        str: 文件扩展名（包含点号），如果无法获取则返回.mp4
    """
    # 尝试从URL路径中提取文件名
    parsed_url = urlparse(url)
    path = parsed_url.path

    # 使用正则表达式匹配文件扩展名
    ext_match = re.search(r'\.(\w+)(?:[?#].*)?$', path)
    if ext_match:
        extension = f'.{ext_match.group(1).lower()}'
        # 验证是否为支持的视频格式
        if extension in VALID_VIDEO_EXTENSIONS:
            return extension

    return '.mp4'  # 默认使用.mp4扩展名


def get_safe_filename(url: str) -> str:
    """生成安全的文件名
    
    使用URL和时间戳的组合生成MD5哈希值作为文件名，确保文件名唯一且简短
    """
    timestamp = str(int(time.time()))
    return hashlib.md5((url + timestamp).encode()).hexdigest()


def check_disk_space(required_bytes: int) -> bool:
    """检查是否有足够的磁盘空间"""
    free_space = shutil.disk_usage(download_dir).free
    return free_space > required_bytes * 1.5  # 预留50%的余量


def download_file(url: str) -> str:
    """从URL下载视频到本地

    Args:
        url (str): 视频URL地址

    Returns:
        str: 下载后的本地文件路径

    Raises:
        HTTPException: 下载失败时抛出异常
    """
    logger = get_logger()
    logger.info(f"开始下载视频: {url}")
    temp_file: Optional[str] = None

    try:
        # 生成唯一的文件名，使用原始URL的文件扩展名
        file_name = f"{get_safe_filename(url)}{get_file_extension(url)}"
        file_path = download_dir / file_name
        temp_file = str(file_path) + '.tmp'

        # 设置请求头和超时
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers)
        response.raise_for_status()

        # 获取文件大小
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件大小超过限制: {MAX_FILE_SIZE / (1024 * 1024)}MB")

        # 检查磁盘空间
        if content_length and not check_disk_space(int(content_length)):
            raise HTTPException(status_code=507, detail="磁盘空间不足")

        # 保存到临时文件
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # 重命名为最终文件
        os.replace(temp_file, file_path)
        logger.info(f"视频下载完成: {file_path}")
        return str(file_path)

    except requests.Timeout:
        raise HTTPException(status_code=408, detail="下载超时")
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"远程文件不存在: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频下载失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")
