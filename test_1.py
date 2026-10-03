"""
服务连接测试脚本
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def test_milvus():
    """测试 Milvus 连接"""
    print("测试 Milvus 连接...")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=os.getenv("MILVUS_URL", "http://192.168.20.101:19530"),
            token=f"{os.getenv('MILVUS_USER', 'root')}:{os.getenv('MILVUS_PASSWORD', 'Milvus')}")
        version = client.get_server_version()
        print(f"  ✓ Milvus 连接成功，版本: {version}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ Milvus 连接失败: {e}")
        return False


def test_mongodb():
    """测试 MongoDB 连接"""
    print("测试 MongoDB 连接...")
    try:
        from pymongo import MongoClient
        client = MongoClient(
            os.getenv("MONGO_URL", "mongodb://192.168.20.101:27017"),
            serverSelectionTimeoutMS=5000,
            #username=os.getenv("MONGO_USER", "admin"),  # 新增：读取账号
            #password=os.getenv("MONGO_PASSWORD", "123456"),  # 新增：读取密码
        )
        # 触发实际连接
        client.admin.command('ping')
        db_names = client.list_database_names()
        print(f"  ✓ MongoDB 连接成功，数据库列表: {db_names}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ MongoDB 连接失败: {e}")
        return False


def test_minio():
    """测试 MinIO 连接"""
    print("测试 MinIO 连接...")
    try:
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "192.168.20.101:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
        buckets = client.list_buckets()
        bucket_names = [b.name for b in buckets]
        print(f"  ✓ MinIO 连接成功，存储桶: {bucket_names}")
        return True
    except Exception as e:
        print(f"  ✗ MinIO 连接失败: {e}")
        return False


def main():
    print("=" * 50)
    print("掌柜智库 - 服务连接测试")
    print("=" * 50)

    results = {
        "Milvus": test_milvus(),
        "MongoDB": test_mongodb(),
        "MinIO": test_minio(),
    }

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    all_passed = True
    for service, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {service}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("所有服务连接正常！")
    else:
        print("存在服务连接失败，请检查配置。")

    return all_passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)