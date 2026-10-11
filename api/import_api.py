"""
导入文件的api

1.创建fastapi实例
2.注册路由
3.利用uvicorn服务器启动fastapi
"""
import os.path

import uvicorn
from fastapi.responses import FileResponse
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.params import Depends
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from knowledge.core.depend import get_import_file_service
from knowledge.core.paths import get_front_page_static_dir
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse
from knowledge.services.import_file_service import ImportFileService
from knowledge.utils.task_util import get_task_info


# 1.创建fastapi实例
def create_app():
    app = FastAPI()

    # 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # 路由注册
    register_router(app)

    # 挂载前端静态资源
    pass


    return app

# 2.注册路由
def register_router(app):

    @app.get("/")
    async def root():
        """根路径重定向到导入页,避免裸 404"""
        return RedirectResponse(url="/import.html")

    @app.get("/import.html")
    async def import_root():
        """返回导入界面"""
        return FileResponse(path=os.path.join(get_front_page_static_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            backGround_tasks: BackgroundTasks,
            importFileService: ImportFileService = Depends(get_import_file_service),
            file: UploadFile = File(...)
    ):
        """
        Post请求处理  接收上传文件
        async def 异步函数，支持高并发，不阻塞主线程
        file: UploadFile 参数类型注解，UploadFile 是 FastAPI 提供的文件对象，包含文件名、内容等
        = File(...) 告诉 FastAPI 这个参数来自请求体中的文件字段（... 表示必填）
        """
        # 1. 上传文件 （本地 + MinIO）
        task_id, file_dir, import_file_path = importFileService.process_upload_file(file)

        # 2. 将耗时的图谱流程放入后台任务，异步处理
        backGround_tasks.add_task(importFileService.run_import_graph, task_id, file_dir, import_file_path)

        # return {
        #     "task_id": "abc-123"
        # }
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """
        Get 请求处理    路径参数自动注入"
        return {
            "status": "processing",
            "done_list": ["上传文件", "检查文件", "PDF转Markdown"],
            "running_list": ["文档切分"],
            "durations": {
                "检查文件": 0.01,
                "PDF转Markdown": 58.23,
                "文档切分": 2.15
            }
        }
        """

        task_info = get_task_info(task_id=task_id)
        return TaskStatusResponse(**task_info)

# 3.利用uvicorn服务器启动fastapi
if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="localhost", port=8000, log_level="info")

