import asyncio
import os.path

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

from knowledge.core.depend import get_query_service
from knowledge.core.paths import get_front_page_static_dir
from knowledge.schema.query_schema import QueryRequest, QueryResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(title="查询服务器", description="知识库查询服务")

    # 解决跨域问题
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带请求头
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )

    # 注册路由
    register_routers(app)

    # 前端静态资源
    front_page_dir = get_front_page_static_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))


    return app


def register_routers(app: FastAPI):

    @app.get("/")
    async def root():
        """根路径重定向到导入页,避免裸 404"""
        return RedirectResponse(url="/chat.html")

    @app.get("/chat.html")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_static_dir(), "chat.html"))

    @app.post("/query")
    async def query(request: QueryRequest,
                    service: QueryService = Depends(get_query_service)):
        # 1.获取session_id
        # request 为什么可.session_id,因为pydantic 会把这些类上的字段声明处理成实例属性，
        # 构造时逐字段校验类型并赋值。所以 request.session_id 就是普通对象属性访问
        session_id = request.session_id or service.generate_session_id()

        # 2.获取任务Id
        task_id = service.generate_task_id()

        # 3.开启流式：提交与订阅合并为一个请求，直接返回 SSE 流
        if request.is_stream:
            # 3.1 必须在图开跑前建好流（且只能在事件循环里建），否则先推的事件会丢
            create_sse_queue(task_id)

            # 3.2 同步图丢线程池，不阻塞事件循环
            asyncio.get_running_loop().run_in_executor(
                None, service.run_query_graph, session_id, task_id, request.query, True
            )

            # 3.3 直接返回流（不再返回 task_id、不再需要客户端二次请求订阅）
            return StreamingResponse(
                sse_generator(task_id),
                media_type="text/event-stream",
                # 禁用各级缓冲，保证 delta 逐条下发（否则经 nginx 会被整段攒住）
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 4.非流式
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, service.run_query_graph, session_id, task_id, request.query, False)

        # 5.获取答案
        answer = service.get_answer(task_id)

        # 6.返回答案
        return QueryResponse(message="处理完成", session_id=session_id, answer=answer,task_id=task_id)

    @app.get("/history/{session_id}")
    async def get_history(session_id: str, limit: int = 50, service: QueryService = Depends(get_query_service)):
        try:
            history_items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": history_items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取历史记录异常：str(e)")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(session_id: str, service: QueryService = Depends(get_query_service)):
        count = service.clear_history(session_id)
        return {"message": "历史记录清除成功", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="localhost", port=8001, log_level="info")
