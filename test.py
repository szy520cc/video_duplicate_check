import os
import argparse
import pickle
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from app.services.video_duplicate_checker import VideoDuplicateChecker

load_dotenv()

# 初始化视频查重器
checker = VideoDuplicateChecker()


def main():

    print("查重系统初始化...")
    # 初始化查重系统，使用命令行参数设置的阈值

    # 处理视频文件夹中的所有视频
    video_dir = Path("video")
    if not video_dir.exists():
        print(f"视频目录不存在: {video_dir}")
        return

    for video_file in video_dir.glob("*.mp4"):
        print(f"\n处理视频: {video_file}")
        is_duplicate, duplicate_info, similarity = checker.check_video(video_file)
        # 如果视频不重复，添加到数据库
        if not is_duplicate:
            # 生成一个基于时间戳的video_id
            video_id = int(time.time())
            # 使用默认的industry_code
            industry_code = str(video_file)
            checker.add_video(video_file, video_id, industry_code)

        # 处理duplicate_info，它可能是JSON字符串、直接的ID或None
        duplicate_id = 0
        if is_duplicate and duplicate_info is not None:
            if isinstance(duplicate_info, str):
                try:
                    # 尝试解析JSON字符串
                    info = json.loads(duplicate_info)
                    duplicate_id = info.get('video_id', 0)
                    industry_code = info.get('industry_code', '')
                except json.JSONDecodeError:
                    # 如果不是JSON字符串，直接使用原值
                    duplicate_id = int(duplicate_info)
            else:
                duplicate_id = int(duplicate_info)

        data = {
            "is_duplicate": is_duplicate,
            "similarity": float(similarity) if similarity else 0.0,
            "video_id": duplicate_id if is_duplicate else None,
            "industry_code": industry_code if is_duplicate else None
        }
        print(f"视频查重结果: {data}")


if __name__ == "__main__":
    main()
