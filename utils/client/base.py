"""
客户端管理器基类，提供：
_require_env(): 环境变量检验
_get_or_create(): 双重检查锁模板方法
"""
import os
import threading


class BaseClientManager:

    @staticmethod
    def _require_env(key: str) -> str:
        """
        读取必须的环境变量，缺失立即抛异常
        os.environ      = 一本完整的通讯录（所有键值对都在里面）
        os.getenv(key)  = 按名字查一个人（一次查一个）
        _require_env    = 按名字查，查不到就报警（必需变量专用）
        :param key:
        :return:
        """
        value = os.getenv(key)  # 完全等价于 value = os.environ.get(key)
        if not value:
            raise EnvironmentError(f"缺少必须的环境变量：{key}")
        return value

    @classmethod
    def _get_or_create(cls, attr_name: str, lock: threading.Lock, factory):
        """
        双重检查锁模板方法。
        factory是一个工厂方法对象（不加括号传入），只有确认需要创建时才调用。
        这就是延迟执行 ---  把“创建”这个动作延迟到真正需要的那一刻。
        :param attr_name:
        :param lock:
        :param factory:
        :return:
        """
        # 第一次检查（无锁，快速返回）
        instance = getattr(cls, attr_name, None)
        """
           ① getattr(cls, attr_name, None)
           Python 反射函数：按字符串名字读取属性。三个参数 =（对象，属性名字符串，找不到时的默认值）。
           getattr(cls, "_openai_client", None)
           # 等价于 cls._openai_client
           # 区别：属性不存在时，直接访问会抛 AttributeError；getattr 带默认值则返回 None
        """
        if instance is not None:
            return instance

        with lock:
            """
            ③ with lock:
            上下文管理器，等价于：
            lock.acquire()      # 上锁：一次只放一个线程进来
            try:
                ... 里面 ...
            finally:
                lock.release()  # 保证无论成败都释放锁
            """
            # 第二次检查（有锁，防止并发重复创建）
            instance = getattr(cls, attr_name, None)

            if instance is not None:
                return instance
            instance = factory()
            """
            factory 是函数对象，加上括号就是执行它
            这就是「延迟执行」——创建动作被推迟到「确认真的需要建」的那一刻。
            """
            setattr(cls, attr_name, instance)   # 等价于 cls._openai_client = instance
            """
            setattr(cls, attr_name, instance)
            getattr 的反操作：按字符串名字写入属性。
            """
            return instance
"""
双重检查锁模板方法。
要解决的问题：多个线程同时第一次要客户端
get_openai() 可能被多个线程同时调用（比如流程并发处理时）。如果客户端还没创建，大家都想创建——怎么办？
只检查一次：两个线程可能同时看到「还没有」，于是各创建一个 → 创建了两个客户端 ❌
每次调用都加锁：能防重复，但每次取已存在的客户端也要排队加锁 → 慢 ❌
双重检查锁 = 「门口快速瞄一眼 + 进去再确认一次」，又快又不会重复创建。
"""







