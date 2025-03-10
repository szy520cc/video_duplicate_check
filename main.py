from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.routers import check_video
import os

app = FastAPI(title="视频查重系统", description="基于多特征融合的视频相似度查重系统", version="1.0.0")


# 全局注册请求验证异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    field_msg = {
        "video_id": "[video_id] 视频ID必填且必须是整数",
        "url": "[url] 视频URL必填且不能为空",
        "industry_code": "[industry_code] 行业code必填且不能为空"
    }
    msg = "参数验证失败: " + str(exc)
    # 检查是否为缺少url字段的错误
    for error in exc.errors():
        if error.get("type") == "missing" and len(error.get("loc", [])) > 1:
            field = error["loc"][1]
            if field in field_msg:
                msg = "参数验证失败: " + field_msg[field]
                break
    # 其他验证错误保持原有格式
    return JSONResponse(
        status_code=422,
        content={
            "status": False,
            "msg": msg
        }
    )


app.include_router(check_video.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")  # 默认主机为0.0.0.0
    port = int(os.getenv("API_PORT", "8811"))  # 默认端口为8811
    uvicorn.run(app, host=host, port=port)
