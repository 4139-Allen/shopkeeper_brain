import logging, os, uuid, shutil
from datetime import datetime

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import kb_import_graph_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, update_task_status

logger = logging.getLogger(__name__)




class ImportFileService:

    @staticmethod
    def _get_data_dir():
        """
        获取上传文件的路径及时间
        "%Y%m%d"  年  月  日
        %Y 四位    %y  两位
        """
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def process_upload_file(self, file) -> tuple[str, str, str]:
        """
            处理上传文件：
                1.生成task_id,构建归档目录
                2.标记upload_file节点为运行中
                3.保存文件到本地磁盘 （:MinIO 管"归档",本地管"干活"）
                4.同步上传到MinIO
                5.标记upload_file节点完成
                6.返回task_id,file_dir,import_file_path

        :param file: 上文件
        :return: Tuple[str,str,str]
        """
        # 1.生成task_id，构建归档目录
        date_dir = self._get_data_dir()
        task_id = str(uuid.uuid4().hex[:8])     #生成一段 8 位随机十六进制字符串，用作简易唯一 ID
        file_dir = os.path.join(date_dir, task_id)

        # 2.标记upload_file节点为运行中
        add_running_task(task_id, "upload_file")

        # 3.保存文件到本地磁盘
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # 4.同步上传到minio
        self._save_pload_file_to_minio(import_file_path, file.filename)

        # 5.标记upload_file节点完成
        add_done_task(task_id, "upload_file")

        # 6.返回task_id,file_dir,import_file_path
        return task_id, file_dir, import_file_path
    
    @staticmethod
    def _save_upload_file_to_local(file, file_dir):
        """保存上传文件到本地目录"""
        # 1.创建文件保存目录
        # 递归创建多级文件夹（父目录不存在也会一起建）
        os.makedirs(file_dir, exist_ok=True)        # 如果文件夹已经存在，不抛异常，直接跳过

        # 2.构建导入文件的路径
        import_file_dir = os.path.join(file_dir, file.filename)

        # 3.写入
        try:
            # "wb"：以二进制写入模式打开文件。
            with open(import_file_dir, "wb") as f:
            # 不同的操作系统以及不同python版本都可以分批次的写入，缓冲区大小： 默认分块大小是 64KB
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败，原因：{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败，原因：{str(e)}")

        # 返回导入的文件完整路径
        return import_file_dir

    @staticmethod
    def _save_pload_file_to_minio(import_file_path, filename):
        """上传文件到minio客户端"""
        # 1.获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"获取minio客户端失败，原因：{str(e)}")
            return

        # 2.获取minio相关信息
        bucket_name = os.getenv("MINIO_BUCKET_NAME", "")
        object_name = f"origin_file/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3.上传文件
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传到minio失败,原因：{str(e)}")

    @staticmethod
    def run_import_graph(task_id: str, file_dir: str, import_file_path: str):
        """运行导入  LangGraph 流水线（后台异步执行任务）"""
        try:
            update_task_status(task_id, "processing")
            global_graph_state = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }
            for event in kb_import_graph_app.stream(global_graph_state):
                for node_name, state in event.items():
                    print(f"[{task_id}] Completed Node: {node_name}")

            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")
            print(f"任务{task_id}执行失败，原因：{str(e)}")













