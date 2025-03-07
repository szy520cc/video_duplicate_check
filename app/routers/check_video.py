# app/routers/check_video.py
import os
import shutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from app.services.video_duplicate_checker import VideoDuplicateChecker
from app.services.download import download_file, check_file_size

router = APIRouter(prefix="/video", tags=["Video"])

# 定义请求模型
class VideoCheckRequest(BaseModel):
    url: str

# 初始化视频查重器
checker = VideoDuplicateChecker()

@router.post("/check")
async def check(request: VideoCheckRequest):
    """检查视频是否重复

    Args:
        request (VideoCheckRequest): 包含视频URL的请求体

    Returns:
        dict: 包含查重结果的响应
    """
    try:
        # 下载视频
        video_path = download_file(request.url)
        try:
            # 检查文件大小
            check_file_size(video_path)

            # 检查视频是否重复
            is_duplicate, duplicate_id, similarity = checker.check_video(video_path)

            # 如果视频不重复，添加到数据库
            if not is_duplicate:
                checker.add_video(video_path)

            return {
                "status": True,
                "msg": "OK",
                "data": {
                    "is_duplicate": is_duplicate,
                    "similarity": float(similarity) if similarity else 0.0,
                    "duplicate_id": int(duplicate_id) if is_duplicate else 0
                }
            }
        finally:
            # 清理下载的视频文件
            if os.path.exists(video_path):
                os.remove(video_path)
    except Exception as e:
        return {"status": False, "msg": str(e)}
