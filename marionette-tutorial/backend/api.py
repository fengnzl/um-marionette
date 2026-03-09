"""
Marionette FastAPI 后端服务

提供轨迹生成和评估的 RESTful API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import IntEnum
import time
import uuid
import torch

# ============================================================================
# 数据模型
# ============================================================================

class DayOfWeek(IntEnum):
    """星期枚举"""
    MONDAY = 25
    TUESDAY = 26
    WEDNESDAY = 27
    THURSDAY = 28
    FRIDAY = 29
    SATURDAY = 30
    SUNDAY = 31


class GenerateRequest(BaseModel):
    """轨迹生成请求"""
    condition1: DayOfWeek = Field(
        ...,
        description="星期几 (25=周一, 26=周二, ..., 31=周日)"
    )
    condition2: int = Field(
        32,
        ge=32, le=33,
        description="斋月状态 (32=非斋月, 33=斋月)"
    )
    condition3: int = Field(
        34,
        ge=34, le=35,
        description="假期状态 (34=非假期, 35=假期)"
    )
    num_samples: int = Field(
        10,
        ge=1, le=100,
        description="生成轨迹数量"
    )
    tmax: float = Field(
        86400.0,
        description="最大时间（秒），默认24小时"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "condition1": 25,
                "condition2": 32,
                "condition3": 34,
                "num_samples": 10,
                "tmax": 86400.0
            }
        }


class Coordinate(BaseModel):
    """GPS 坐标"""
    lat: float = Field(..., ge=-90, le=90, description="纬度")
    lon: float = Field(..., ge=-180, le=180, description="经度")


class TrajectoryPoint(BaseModel):
    """轨迹点"""
    time: float = Field(..., description="到达时间（秒）")
    poi_id: int = Field(..., description="POI ID")
    category: int = Field(..., description="POI 类别")
    coordinate: Optional[Coordinate] = None


class GeneratedTrajectory(BaseModel):
    """生成的轨迹"""
    sequence_id: str = Field(..., description="序列唯一标识")
    points: List[TrajectoryPoint] = Field(..., description="轨迹点列表")
    conditions: Dict = Field(..., description="生成条件")


class GenerateResponse(BaseModel):
    """生成响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    sequences: List[GeneratedTrajectory] = Field(..., description="生成的轨迹列表")
    generation_time: float = Field(..., description="生成耗时（秒）")


class ModelInfo(BaseModel):
    """模型信息"""
    id: str = Field(..., description="模型 ID")
    name: str = Field(..., description="模型名称")
    description: str = Field(..., description="模型描述")
    loaded: bool = Field(..., description="是否已加载")
    parameters: Dict = Field(..., description="模型参数")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    gpu_available: bool = Field(..., description="GPU 是否可用")


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="Marionette API",
    description="轨迹生成服务 API - 基于扩散模型的时空数据生成",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型缓存
models_cache: Dict[str, any] = {}
gps_dict: Dict = {}

# ============================================================================
# 辅助函数
# ============================================================================

def create_batch_from_request(request: GenerateRequest):
    """
    从请求创建批次数据

    TODO: 实现完整的批次创建逻辑
    """
    # 这是一个简化的实现
    # 实际需要根据 datamodule.py 的 Batch 类来实现
    batch = {
        "condition1": [request.condition1] * request.num_samples,
        "condition2": [request.condition2] * request.num_samples,
        "condition3": [request.condition3] * request.num_samples,
        "tmax": [request.tmax] * request.num_samples,
        "batch_size": request.num_samples
    }
    return batch


def convert_to_json_format(generated_batch, gps_dict):
    """
    将生成的批次转换为 JSON 格式

    TODO: 实现完整的格式转换逻辑
    """
    sequences = []
    for i in range(generated_batch.get("batch_size", 0)):
        seq_id = str(uuid.uuid4())
        points = []

        # TODO: 从 generated_batch 提取轨迹点
        # 这里需要根据实际的生成结果格式来实现

        sequences.append({
            "sequence_id": seq_id,
            "points": points,
            "conditions": {
                "day_of_week": generated_batch.get("condition1", [])[i] if i < len(generated_batch.get("condition1", [])) else None,
                "ramadan": generated_batch.get("condition2", [])[i] if i < len(generated_batch.get("condition2", [])) else None,
                "holiday": generated_batch.get("condition3", [])[i] if i < len(generated_batch.get("condition3", [])) else None,
            }
        })

    return sequences


# ============================================================================
# API 端点
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """根端点"""
    return {
        "message": "Welcome to Marionette API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    gpu_available = torch.cuda.is_available()
    model_loaded = "marionette" in models_cache

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        gpu_available=gpu_available
    )


@app.get("/api/v1/models", response_model=List[ModelInfo])
async def get_models():
    """
    获取可用模型列表
    """
    models = [
        ModelInfo(
            id="marionette",
            name="Marionette",
            description="细粒度可控轨迹生成模型 (KDD'25)",
            loaded="marionette" in models_cache,
            parameters={
                "temporal_steps": 100,
                "spatial_steps": 200,
                "num_condition_types": 6
            }
        )
    ]
    return models


@app.post("/api/v1/models/{model_id}/load")
async def load_model(model_id: str):
    """
    加载模型到内存

    TODO: 实现实际的模型加载逻辑
    """
    if model_id in models_cache:
        return {"status": "already_loaded", "model_id": model_id}

    try:
        # TODO: 实际的模型加载代码
        # checkpoint_path = f"checkpoints/{model_id}.ckpt"
        # task = DensityEstimation.load_from_checkpoint(checkpoint_path)
        # task.eval()
        # models_cache[model_id] = task

        # 模拟加载成功
        models_cache[model_id] = {"loaded": True}

        return {"status": "loaded", "model_id": model_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型加载失败: {str(e)}"
        )


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_trajectories(request: GenerateRequest):
    """
    生成轨迹

    流程:
    1. 验证请求参数
    2. 检查/加载模型
    3. 构造输入批次
    4. 运行模型推理
    5. 转换输出格式
    6. 返回结果
    """
    start_time = time.time()

    try:
        # 1. 检查模型
        model_key = "marionette"
        if model_key not in models_cache:
            # 自动加载模型
            await load_model(model_key)

        task = models_cache[model_key]

        # 2. 构造批次
        batch = create_batch_from_request(request)

        # TODO: 3. 生成时间样本
        # time_samples = task.tpp_model.sample(
        #     batch_size=request.num_samples,
        #     tmax=request.tmax,
        #     x_n=batch
        # )

        # TODO: 4. 生成 POI 序列
        # generated = task.discrete_diffusion.sample_fast(time_samples)

        # TODO: 5. 转换为 JSON 格式
        # sequences = convert_to_json_format(generated, gps_dict)

        # 模拟生成结果（实际应该从模型获取）
        sequences = []
        for i in range(request.num_samples):
            seq_id = str(uuid.uuid4())
            sequences.append(GeneratedTrajectory(
                sequence_id=seq_id,
                points=[
                    TrajectoryPoint(
                        time=3600 * (i + 1),
                        poi_id=10 + i,
                        category=i % 9,
                        coordinate=Coordinate(lat=41.0082, lon=28.9784)
                    )
                ],
                conditions={
                    "day_of_week": request.condition1,
                    "ramadan": request.condition2,
                    "holiday": request.condition3
                }
            ))

        generation_time = time.time() - start_time

        return GenerateResponse(
            success=True,
            message=f"成功生成 {len(sequences)} 条轨迹",
            sequences=sequences,
            generation_time=generation_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成失败: {str(e)}"
        )


@app.post("/api/v1/evaluate")
async def run_evaluation(task: str = "LocRec"):
    """
    运行评估任务

    TODO: 实现实际的评估逻辑
    """
    available_tasks = ["LocRec", "NexLoc", "SemLoc", "Stat", "EpiSim"]

    if task not in available_tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"未知任务: {task}。可用任务: {', '.join(available_tasks)}"
        )

    # TODO: 实际运行评估
    # results = run_evaluation_task(task)

    return {
        "task": task,
        "status": "completed",
        "results": {
            "message": "评估功能待实现"
        }
    }


# ============================================================================
# 启动命令
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Marionette API Server")
    print("=" * 60)
    print(f"API 文档: http://localhost:8000/docs")
    print(f"健康检查: http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
