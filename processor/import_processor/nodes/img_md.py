"""
md文档图片处理节点

将 MarkDownImageNode 中逻辑拆分为四个职责单一的协作类，统一调度
"""
import base64
import dataclasses
import re
import time
from collections import deque
from pathlib import Path
from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


# 1.文件读写 & 备份
class MdFileHandler:
    """
    主要职责：
        1.读取md内容、md_path、图片日志
        2.备份新的md_content（方便测试观看）
    """
    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    # ①读取
    def read_md(self, state:ImportGraphState) -> tuple[str, Path, Path]:

        # Path("").exists()   # → True！（当前目录），需要单独判断是否为空
        md_path = state.get("md_path", "")
        if not md_path:                     #如果为空，状态字段抛出异常
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)
        # 标准化路径处理
        md_path_obj = Path(md_path)
        # 判断磁盘上是否存在该文件
        if not md_path_obj:
            raise FileProcessingError(f"md文件路径无效：{md_path}", node_name=self.node_name)
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        image_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, image_dir

    # ②备份
    def backup_md(self,md_path_obj:Path, new_md_content:str) -> str:
        self.logger.info(f"正在备份md文件：{md_path_obj}")
        #拼接备份文档路径
        new_md_file_path = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        try:    #写入新路径
            with open(new_md_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"备份成功：{new_md_file_path}")
        except Exception as e:
            self.logger.info(f"备份失败：{new_md_file_path}: {e}")
            raise ImageProcessingError(message=f"md文档备份失败：{new_md_file_path}:{e}", node_name=self.node_name)

        return str(new_md_file_path)


# 2.图片扫描 & 图片上下文提取
# 2.1辅助工具，定义数据类型
# 未来直接实例化，不用写__init__  __repr__方法
@dataclasses.dataclass
class ImageContext:
    """图片的md文档的上下文内容信息"""
    heading:str     #离图片上方最近的标题
    pre_text:str    #图片上方的正文内容
    post_text:str   #图片下方的正文内容

@dataclasses.dataclass
class ImageInfo:
    """一张图片完整信息"""
    name:str                #图片文件名，如abc.jpg，作为存储图片摘要的字典容器key
    path:str                #图片完整路径，如/home/user/abc.jpg，供vlm和minio使用
    context:ImageContext    #图片在md文档中的上下文信息，供vlm使用（阅读上下文提取对应图片）

# 2.2主要职责逻辑实现
class ImageScanner:
    """
    主要职责：
        1.根据存放图片目录，得到该目录下的有效图片文件，目录和非图片排除
        2.去md文件中找到图片的位置
        3.获取改图片的上下文内容，给VLM模型提供图片上下文信息，帮助模型识别图片更加准确
        4.最终组装所有图片的上下文内容（list）
    """
    def __init__(self, logger):
        self.logger = logger

    # ①扫描图片,返回图片与图片上下文信息列表
    def scan_img_dir(self,
                     img_dir:Path,
                     md_content: str,
                     image_extensions: set[str],
                     context_length: int
                     ) -> list[ImageInfo]:
        """
        :param img_dir: 图片目录
        :param md_content: MD内容
        :param image_extensions: 扩展名
        :param context_length: 上下文的长度，各自最大长度不能超过200
        :return: List[ImageInfo]
        """
        image_list: list[ImageInfo] = []
        # 1.扫描图片目录下所有文件
        for img_path in Path(img_dir).iterdir():    # Path(img_dir).iterdir() -> 迭代器，遍历图片所在目录
            if not img_path.is_file():              # 判断这个路径指向的是不是一个「普通文件」（而不是目录、快捷方式等）
                self.logger.warning(f"图片目录下存在非文件，{img_path}")
                continue    # 结束本次循环，之后循环继续，处理下一个条目（相当于非文件跳过）
            if img_path.suffix.lower() not in image_extensions:
                self.logger.warning(f"图片目录下存在不支持的文件，请留意：{img_path}")
                continue
            # 2.去md文档中找到图片出现位置的上下文，找不到返回None
            img_context = self._find_context(md_content, img_path.name, context_length)
            if img_context is None:
                self.logger.warning(f"md文件中未找到图片：{img_path.name}的引用")
                continue
            # 3.找到并存入图片信息列表中
            image_list.append(ImageInfo(img_path.name, str(img_path), img_context))
        self.logger.info(f"找到了{len(image_list)}张图片")
        # 返回图片信息列表
        return image_list

    def _find_context(self, md_content: str, img_name: str, max_chars: int = 200) -> ImageContext | None:
        """找图片上下文方法"""
        # 正则模型
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        # 按行切
        md_lines = md_content.split("\n")
        # 遍历找到图片索引
        for line_idx, line in enumerate(md_lines):   # (序号，元素)元组
            if not pattern.search(line):             #切完后进行逐行搜索匹配图片对象
                continue                             #没找到，下一轮继续找
            # 匹配到图片索引
            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._fing_heading_above(md_lines, line_idx)
            pre_content = md_lines[prev_boundary + 1: line_idx]  # 切片[标题所在行＋1:图片所在行]
            img_pre = self._extract_limited_context(pre_content, max_chars, direction="front")

            # 向下：找下一个标题，取图片到标题之间的内容作用下文
            next_boundary = self._find_heading_below(md_lines, line_idx)
            post_content = md_lines[line_idx + 1: next_boundary]
            img_post = self._extract_limited_context(post_content, max_chars, "end")

            # 返回上下文信息
            return ImageContext(
                heading=prev_title,     # 标题
                pre_text=img_pre,       # 上文
                post_text=img_post      # 下文
            )
        return None


    @staticmethod
    def _fing_heading_above(md_lines: list[str], line_idx: int) -> tuple[str, int]:
        """
        从图片位置向上查找标题
        :param md_lines:按行切片后的md所有内容列表
        :param line_idx:图片所在行号
        :return:(图片上方最近的标题名字，对应的行号)元组
        """
        # 从图片所在行的上一行开始往上遍历
        for i in range(line_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):    #匹配到标题及其所在行
                return md_lines[i], i
        return "", -1   # 没找到标题：说明图片是文档第一段

    @staticmethod
    def _find_heading_below(md_lines: list[str], line_idx: int) -> int:
        """
        从图片位置向下查找下一个标题
        :param md_lines:按行切片后的md所有内容列表
        :param line_idx:图片所在行号
        :return:
        """
        for i in range(line_idx + 1, len(md_lines), 1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):    #匹配到标题及其所在行
                return i
        return len(md_lines)    #找不到标题：说明图片是文档最后一段

    @staticmethod
    def _extract_limited_context(pre_content: list[str], max_chars: int, direction: str) -> str:
        """按段落分割，按direction方向贪心装填，保持段落完整性"""
        """
        这个函数用于从Markdown文本中提取有限长度的上下文内容,保持段落完整性。
            工作流程:
            1. 段落分割阶段
            遍历所有行,识别空白行或其他图片标记(![...](...))作为段落分隔符
            将连续的非空行组合成一个段落
            所有段落存入 paragraphs 列表
            2. 方向处理
            direction == "front": 向前提取(图片上方的内容),需要反转段落顺序(反转原理：离图片近的相关性更高)
            其他方向: 向后提取(图片下方的内容),保持原顺序
            3. 贪心装填阶段
            按段落顺序累加,直到总字符数超过 max_chars
            保证段落完整性,不会截断段落
            如果已有选中段落且超出限制,则停止添加
            4. 结果组装
            如果是向前提取,再次反转选中的段落恢复正确顺序
            用双换行符 \n\n 连接所有选中段落并返回
        两次反转的作用：
            第一次反转：装填前——「让离图片最近的排前面」
                 目的：优先保住离图片最近的内容。
            第二次反转：装填后——「恢复阅读顺序」
                 目的：输出给 VLM 的内容必须按原文顺序，方便阅读
        """
        current_paragraph: list[str] = []       # 临时存储当前正在构建的的段落的行
        paragraph: list[str] = []               # 存储所有已完成的段落
        # 遍历图片上方所有上文
        for line in pre_content:
            # 是空白行
            is_blank_line = not line.strip()    # 空白行删除，返回False(not False -> Ture)
            # 是其他图片，line.strip() 是为了消除行首尾空白对正则锚点的干扰
            is_other_image = re.match(r"^!\[.*?\]\(.*?\)$", line.strip())
            if is_blank_line or is_other_image:     # 遇到（空行/图片行)
                if current_paragraph:               # 判断容器非空才归档
                    paragraph.append("\n".join(current_paragraph))  # 临时容器的内容，存入主容器（归档）
                    current_paragraph = []          # 清理临时容器
                continue
            # 正常内容存入 临时容器
            current_paragraph.append(line)
        # for 循环跑完后，临时容器里还攒着最后一个未归档的段落（情景:最后一行不是图片或空行）。
        if current_paragraph:
            paragraph.append("\n".join(current_paragraph))
        if direction == "front":
            paragraph.reverse()     # 反转段落顺序，就近优先

        total = 0                   # 累计字符数
        selected: list[str] = []    # 存储选中的段落

        # 贪心装填
        for para in paragraph:
            if total + len(para) > max_chars and selected:  # 同时满足跳出循环（有内容时，保证至少有一个段落存入）
                break
            selected.append(para)
            total += len(para)
        # 回复原来顺序，便于大模型阅读
        if direction == "front":
            selected.reverse()
        # 返回这张图片上下文
        return "\n\n".join(selected)

# 3.图片摘要生成
class VLMSummarizer:
    """
    主要职责:
        根据图片上下文和图片，靠VLM视觉语言模型生成图片摘要
    """
    def __init__(self, logger):
        self.logger = logger


    def summarize_all(self,
                       document_title: str,
                       image_list: list[ImageInfo],
                       vl_model: str,
                       requests_per_minute: int # 每分钟请求次数
                       ) -> dict[str, str]:     # 返回图片名称 和 图片摘要 字典

        summaries: dict[str, str] = {}
        requests_timestamps: deque[float] = deque() # 请求时间戳

        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(f"VLM不可用，跳过图片摘要生成：{e}")
            for img in image_list:
                summaries[img.name] = "图片描述默认值"
            return summaries

        for img in image_list:
            self._enforce_rate_limit(requests_timestamps, requests_per_minute)  #请求速率限制
            # 调用vlm模型生成图片摘要
            summaries[img.name] = self._summarize_one(client, vl_model, document_title, img)
        self.logger.info(f"生成{len(summaries)}张图片摘要")
        return summaries


    def _enforce_rate_limit(self, timestamps: deque[float], max_requests: int, window: int = 60):
        """
        欢动窗口速率限制算法，用于控制API请求频率
        :param timestamps:   双端队列，存储每次请求的时间戳，Deque支持高效的头部删除和尾部添加操作
        :param max_requests:  最大的请求数量。最多每分钟处理图片摘要的请求数量。超过限制时自动暂停
        :param window:  时间窗口 默认60秒，一分钟。只保留最近60秒内的请求。
        """
        # 获取当前系统时间
        now = time.time()
        # 当前队列不为空，而且队列中的第一个时间戳超过了时间窗口，则删除队列中的第一个时间戳
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()
        # 档期队列长度超过最大请求数量，则暂停一段时间
        if len(timestamps) >= max_requests:
            # 等待时间 = 时间窗口 - (当前时间 - 第一个时间戳)
            sleep_duration = window - (now - timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停{sleep_duration:.2f}秒...")
                time.sleep(sleep_duration)
            # 重新清理并记录
            # 休眠后再次清理过期时间戳
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()
        # 将当前请求时间加入队列
        timestamps.append(now)

    def _summarize_one(self, client: OpenAI, vl_model: str, document_title: str, img: ImageInfo) -> str:
        """
        调用VLM模型，获取图片摘要
        :param client: VLM模型客户端对象
        :param vl_model: 模型名称
        :param document_title: md文件名称
        :param img: ImageInfo  图片信息
        :return: 摘要信息
        """
        # 使用列表推导式从图片上下文的三个属性中筛选出非空字符串
        parts = [p for p in (img.context.heading, img.context.pre_text, img.context.post_text) if p]
        final_context = "\n".join(parts) if parts else "暂无可用上下文"
        try:    #以 "rb" 二进制模式读取图片，得到的是JPEG/PNG 的原始二进制字节流（bytes）
            with open(img.path, "rb") as f:
                # b64encode：二进制图片 → Base64文本形式（解决"二进制进不了 JSON"问题，但结果还是bytes）
                # decode("utf-8")：Base64bytes → Python字符串（解决"bytes 拼不进 f-string 和 JSON"问题）
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"
        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.path}: {e}")
            return "图片描述"


# 4.图片对象上传至MinIo，替换md文档图片地址
class ImageUploader:
    """
    主要职责：
        1.将本地图片上传到MinIo,得到MinIo上图片的访问地址
        2.替换 md图片的地址
    """
    def __init__(self, logger):
        self.logger = logger

    def upload_and_replace(self,
                           document_name,
                           md_content,
                           images_summaries,
                           image_list,
                           minio_bucket,
                           minio_base_url) -> str:
        """
        MinIO上传，正则替换MD内容。返回新的MD文档内容
        :param document_name:  MD文档名称
        :param md_content:  MD文档内容
        :param images_summaries:  图片-摘要
        :param image_list: 图片列表
        :param minio_bucket: 桶
        :param minio_base_url: MinIO的地址
        :return: 新的MD内容
        """
        remote_urls = self._upload_all(document_name, image_list, minio_bucket, minio_base_url)
        new_md_content = self._replace_in_md(md_content, images_summaries, remote_urls)
        return new_md_content

    def _upload_all(self, document_name, image_list: list[ImageInfo], minio_bucket, minio_base_url):
        """上传图片"""
        remote_urls: dict[str, str] = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning(f"MinIO不可用,所有图片保留本地路径：{e}")
            for img in image_list:
                remote_urls[img.name] = img.path
            return remote_urls
        for img in image_list:
            object_name = f"{document_name}/{img.name}"
            try:
                minio_client.fput_object(minio_bucket, object_name, img.path)
                remote_url = f"{minio_base_url}/{minio_bucket}/{object_name}"
                self.logger.info(f"图片上传成功：{img.name}")
                remote_urls[img.name] = remote_url
            except Exception as e:
                self.logger.warning(f"图片上传失败：{img.name},保留本地路径")
                remote_urls[img.name] = img.path

        self.logger.info(f"成功上传{len(remote_urls)}张图片到MinIO")
        return remote_urls

    @staticmethod
    def _replace_in_md(md_content, images_summaries, remote_urls) -> str:
        """替换MD中图片引用为远程URL + 摘要"""
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_path = Path(original_path).name
            for img_name, summery in images_summaries.items():
                if file_name_in_path == img_name:
                    return f"![{summery}]({remote_urls[img_name]})"
            return match.group(0)

        # 使用 pattern.sub() 方法对 md_content 进行替换，对所有匹配的图片调用 replacer 函数，返回处理后的完整 Markdown 文本。
        return pattern.sub(replacer, md_content)

# --定义主节点--
class MarkDownImageNode(BaseNode):
    """md文档处理主节点，编排调度"""

    name = "md_img_node"

    def __init__(self):
        super().__init__()      #调用父类属性（config、logger）
        self.file_handler = MdFileHandler(self.logger, self.name)   #md文件处理
        self.scanner = ImageScanner(self.logger)                    #图片扫描，上下文提取
        self.summarizer = VLMSummarizer(self.logger)                #图像识别模型 识别图片生成摘要
        self.uploader = ImageUploader(self.logger)                  #图片对象上传至minio


    def process(self, state:ImportGraphState):
        """
        处理逻辑，主要职责：
            1.得到四个类的实例对象
            2.分别调用四个实例对象的处理方法
        """
        self.log_step("step1", "读取md文件内容，路径以及图片的目录")
        # 1.读取文件，如果没有图片路径，直接返回内容
        md_content, md_path_obj, image_dir = self.file_handler.read_md(state)
        if not image_dir.exists():
            self.logger.warning(f"文件{md_path_obj.name}暂无图片要处理")
            state["md_content"] = md_content
            return state

        # 2.图片扫描 & 提取图片上下文
        self.log_step("step2", "准备开始扫描图片目录")
        image_list: list[ImageInfo] = self.scanner.scan_img_dir(
            img_dir=image_dir,
            md_content=md_content,
            image_extensions=self.config.image_extensions,
            context_length=self.config.img_content_length
        )

        # 3.VLM生成摘要
        self.log_step("step3", "VLM生成图片摘要")
        summaries = self.summarizer.summarize_all(
            document_title=md_path_obj.name,  # MD文件名称
            image_list=image_list,  # MD中图片列表
            vl_model=self.config.vl_model,  # VLM模型名称
            requests_per_minute=self.config.requests_per_minute  # 每分钟请求次数限制
        )
        print(summaries)
        """
        {'01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）', 
        '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示', 
        '115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg': '中文说明书标识', 
        '347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg': '最大量程接线端子标识', 
        '3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg': '交流电压测量时表笔正确连接至被测电路的示意图', 
        '632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg': '直流电流测量接线示意图（注意10A档位测量时间不超过30秒）', 
        '7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg': '万用表RS-12各部件标识示意图', 
        '9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg': '安全警示标志', 
        '9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg': '高压危险警示符号', 
        'b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg': '安全标识符号示意图', 
        'c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg': 'RS-12数字万用表正面面板结构示意图', 
        'c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg': 'RS PRO品牌标识', 
        'de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg': '万用表电阻测量接线示意图'}
        """
        # 4.图片上传MinIo & 替换
        self.log_step("step4", "上传图片至MinIo，替换md文档图片地址")
        new_md_content = self.uploader.upload_and_replace(
            document_name=md_path_obj.stem,
            md_content=md_content,
            images_summaries=summaries,
            image_list=image_list,
            minio_bucket=self.config.minio_bucket,
            minio_base_url=self.config.get_minio_base_url()
        )

        # 5.备份
        self.file_handler.backup_md(md_path_obj, new_md_content)

        # 6.更新并返回
        state["md_content"] = new_md_content

        return state

#测试
if __name__ == "__main__":
    setup_logging()
    node = MarkDownImageNode()
    state:ImportGraphState = {
        "md_path":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    }
    node.process(state)





