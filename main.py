from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.routers import check_video
import os

app = FastAPI(title="视频查重系统", description="基于多特征融合的视频相似度查重系统", version="1.0.0")


# 全局注册请求验证异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 检查是否为缺少url字段的错误
    for error in exc.errors():
        if error["loc"] == ("body", "url") and error["type"] == "missing":
            return JSONResponse(
                status_code=422,
                content={
                    "status": False,
                    "msg": "url 为必填项且不能为空"
                }
            )
    
    # 其他验证错误保持原有格式
    return JSONResponse(
        status_code=422,
        content={
            "status": False,
            "msg": "参数验证失败: " + str(exc)
        }
    )

app.include_router(check_video.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0") # 默认主机为0.0.0.0
    port = int(os.getenv("API_PORT", "8811"))    # 默认端口为8811
    uvicorn.run(app, host=host, port=port)