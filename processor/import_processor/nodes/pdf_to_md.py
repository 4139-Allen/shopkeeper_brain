"""
pdf转md节点
使用mineru 将pdf文档转换为md文档
"""
import json
import subprocess
import time
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import PdfConversionError, ValidationError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    PDF 转 MD 节点
    """
    name = "pdf_to_md_node"

    def process(self, state:ImportGraphState):
        """
        执行转换流程
        :param state: 图状态
        :return: 更新后的图状态 (包含md_path)
        """

        # 1.校验导入文件的合法性
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2.使用mineru将pdf转换为md
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("pdf转换失败", self.name)

        # 3.获取md 的 path
        md_path = self._get_md_paths(import_file_path, file_dir_path)

        # 4.更新state字典的md_path字段
        state["md_path"] = md_path

        # 5.返回更新后的state
        return state

    def _validate_state_inputs_path(self, state):
        """
        校验导入的文档合法性
        :param state: 该节点从入口节点收到的状态
        :return: (import_file_path_obj,file_dir_path_obj)元组
        """
        self.log_step("step1", "对状态的路径输入参数进行校验")

        # 1.获取pdf文件路径
        import_file_path = state.get("import_file_path", "")

        # 2.获取文档转换后 输出存放的目录
        file_dir_path = state.get("file_dir", "")

        # 3.判断文件是否存在
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 4.使用Path方法标准化路径
        import_file_path_obj = Path(import_file_path)

        # 5.校验是一个真实路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不是真实存在的路径", self.name)

        # 6.判断转换后输出的文件目录是否存在，本项目解析输出的文件跟导入文件同目录
        if not file_dir_path:
            file_dir_path_obj = import_file_path_obj.parent
        else:
            file_dir_path_obj = Path(file_dir_path)

        # 8.打印日志
        self.logger.info(f"需要解析的pdf文档，源路径：{import_file_path_obj}")
        self.logger.info(f"解析后的md文档存放目录：{file_dir_path_obj}")

        return import_file_path_obj, file_dir_path_obj  #返回元组


    def _execute_mineru(self, import_file_path, file_dir_path):
        """
            执行mineru命令将pdf转换为md
            :param import_file_path: 解析的pdf文件全路径
            :param file_dir_path: 解析后md的存储目录
            :return: 命令执行结果状态码  0 表示成功
        """
        self.log_step("step2", "执行mineru解析 PDF to MarkDown")

        # 1.构建命令
        cmd =["mineru",
              "-p",
              str(import_file_path),
              "-o",
              str(file_dir_path),
              "--backend",
              "pipeline"
]
        process_start_time = time.time()

        # 2.执行命令
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,         #子进程执行命令的输出回流
            stderr=subprocess.STDOUT,       #子进程执行命令的错误日志输出回流
            errors="replace",               #遇到乱码时替换
            text=True,                      #输出的内容为字符串，不是二进制字节
            encoding="utf-8",               #指定中文字符串编码
            bufsize=1                       #按行缓存，只要缓冲区一行满了就输出，模拟实时效果
        )

        # 3.打印子进程日志信息
        for line in proc.stdout:
            self.logger.info(f"执行mineru产生的日志：{line}")

        # 4.等待子进程执行完成，获取命令执行结果（0表示成功）
        processed_code = proc.wait()

        process_end_time = time.time()

        # 5.判断是否执行成功    -- Path.name -> 获取文件名字
        if processed_code == 0:
            self.logger.info(f"执行mineru命令解析PDF文件成功：{import_file_path.name}，耗时：{process_end_time - process_start_time:.2f}秒")
        else:
            self.logger.error(f"执行mineru命令解析PDF文件失败：{import_file_path.name}，错误码processed_code：{processed_code}")

        # 6.返回执行结果状态码
        return processed_code

    @staticmethod
    def _get_md_paths(import_file_path, file_dir_path):
        """
           获取PDF解析成MD后的MD文件的实际路径
           :param import_file_path: 被解析的PDF文件路径，全路径(路径+文件名)
           :param file_dir_path: 指定的Pdf解析后的MD存储路径
           :return: 实际的MD文件全路径(路径+文件名)
        """
        # pdf文件的标题(名称)，不含扩展命和路径(Path.stem -> 获取文件名字不含扩展名)
        file_name = import_file_path.stem
        # 拼接md文档路径
        md_path = file_dir_path / file_name / "auto" / f"{file_name}.md"
        return str(md_path)


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()
    init_state:ImportGraphState = {
        "import_file_path":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用.pdf",
        "file_dir":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"

    }
    result = pdf_to_md_node.process(init_state)
    print(json.dumps(result, indent=4, ensure_ascii=False))


