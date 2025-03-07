import os
import argparse
import pickle
import time
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
        is_duplicate, duplicate_id, similarity = checker.check_video(video_file)
        # 如果视频不重复，添加到数据库
        if not is_duplicate:
            checker.add_video(video_file)

        data = {
            "is_duplicate": is_duplicate,
            "similarity": float(similarity) if similarity else 0.0,
            "duplicate_id": int(duplicate_id) if is_duplicate else 0
        }
        print(f"视频查重结果: {data}")


if __name__ == "__main__":
    main()
