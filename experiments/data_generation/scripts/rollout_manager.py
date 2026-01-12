#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollout 管理器

负责：
1. 分批保存 responses 到 JSONL 文件
2. 记录和管理元信息
3. 断点续传支持
4. 可选的 WandB 监控

目录结构:
    rollouts/
        rollout_<timestamp>/
            meta.json                    # 整体元信息（只保存一次）
            checkpoint.json              # 断点续传信息
            responses_00001_01000.jsonl  # 分批保存的 responses
            responses_01001_02000.jsonl
            ...
"""

import os
import json
import time
import glob
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==================== 数据类 ====================

@dataclass
class SamplingParams:
    """采样参数"""
    n_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    max_model_len: int
    max_num_seqs: int
    ignore_eos: bool = False


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    trust_remote_code: bool


@dataclass
class RolloutMeta:
    """Rollout 元信息"""
    # 基本信息
    rollout_id: str
    created_at: str
    
    # 模型配置
    model_config: Dict[str, Any]
    
    # 采样参数
    sampling_params: Dict[str, Any]
    
    # 数据信息
    total_questions: int
    n_samples_per_question: int
    total_responses: int
    
    # 输入文件
    formatted_prompts_path: str
    sampled_questions_path: str
    
    # 保存配置
    save_batch_size: int
    output_dir: str
    
    # 运行时统计（会在运行过程中更新）
    status: str = "running"  # running, completed, interrupted
    completed_questions: int = 0
    elapsed_time_seconds: float = 0.0
    
    # Token 统计
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    
    # 停止原因统计
    finish_reason_counts: Dict[str, int] = field(default_factory=dict)
    
    # 合并后的文件路径（完成时生成）
    merged_file: Optional[str] = None


@dataclass
class Checkpoint:
    """断点续传信息"""
    rollout_id: str
    last_completed_q_id: int  # 最后完成的问题 ID（-1 表示未开始）
    last_saved_batch_end: int  # 最后保存的批次结束位置
    updated_at: str
    
    # 累计统计
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    finish_reason_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ResponseRecord:
    """单条 response 记录"""
    q_id: int
    sample_idx: int
    question: str
    gt_answer: str
    prompt: str  # 格式化后的完整 prompt
    response: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


# ==================== Checkpoint 管理器 ====================

class CheckpointManager:
    """断点续传管理器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.meta_path = self.output_dir / "meta.json"
    
    def load_checkpoint(self) -> Optional[Checkpoint]:
        """加载断点信息"""
        if not self.checkpoint_path.exists():
            return None
        
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
        
        return Checkpoint(**data)
    
    def save_checkpoint(self, checkpoint: Checkpoint):
        """保存断点信息"""
        checkpoint.updated_at = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
    
    def load_meta(self) -> Optional[RolloutMeta]:
        """加载元信息"""
        if not self.meta_path.exists():
            return None
        
        with open(self.meta_path, 'r') as f:
            data = json.load(f)
        
        return RolloutMeta(**data)
    
    def save_meta(self, meta: RolloutMeta):
        """保存元信息"""
        with open(self.meta_path, 'w') as f:
            json.dump(asdict(meta), f, indent=2, ensure_ascii=False)
    
    def get_resume_start_index(self) -> int:
        """获取续传的起始索引"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return 0
        return checkpoint.last_completed_q_id + 1
    
    @staticmethod
    def find_latest_incomplete_rollout(rollout_base_dir: str) -> Optional[str]:
        """查找最近的未完成 rollout"""
        base_dir = Path(rollout_base_dir)
        if not base_dir.exists():
            return None
        
        # 查找所有 rollout 目录
        rollout_dirs = sorted(base_dir.glob("rollout_*"), reverse=True)
        
        for rollout_dir in rollout_dirs:
            meta_path = rollout_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get("status") != "completed":
                    return str(rollout_dir)
        
        return None


# ==================== Response 保存器 ====================

class ResponseSaver:
    """分批保存 responses"""
    
    def __init__(
        self,
        output_dir: str,
        batch_size: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.current_batch: List[ResponseRecord] = []
        self.current_batch_start = 0
        
        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_batch_filename(self, start_q_id: int, end_q_id: int) -> str:
        """生成批次文件名"""
        return f"responses_{start_q_id:05d}_{end_q_id:05d}.jsonl"
    
    def add_responses(
        self,
        q_id: int,
        question: str,
        gt_answer: str,
        prompt: str,
        responses: List[str],
        prompt_tokens: int,
        completion_tokens_list: List[int],
        finish_reasons: List[str],
    ) -> Optional[str]:
        """
        添加一个问题的所有 responses
        
        Returns:
            如果触发保存，返回保存的文件路径；否则返回 None
        """
        for sample_idx, (response, comp_tokens, finish_reason) in enumerate(
            zip(responses, completion_tokens_list, finish_reasons)
        ):
            record = ResponseRecord(
                q_id=q_id,
                sample_idx=sample_idx,
                question=question,
                gt_answer=gt_answer,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                finish_reason=finish_reason,
            )
            self.current_batch.append(record)
        
        # 检查是否需要保存
        questions_in_batch = len(set(r.q_id for r in self.current_batch))
        if questions_in_batch >= self.batch_size:
            return self.flush()
        
        return None
    
    def flush(self) -> Optional[str]:
        """将当前批次写入文件"""
        if not self.current_batch:
            return None
        
        # 计算批次范围
        q_ids = [r.q_id for r in self.current_batch]
        start_q_id = min(q_ids)
        end_q_id = max(q_ids)
        
        # 生成文件名
        filename = self._get_batch_filename(start_q_id, end_q_id)
        filepath = self.output_dir / filename
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in self.current_batch:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
        
        print(f"✓ 已保存批次: {filename} ({len(self.current_batch)} 条记录, 问题 {start_q_id}-{end_q_id})")
        
        # 清空当前批次
        self.current_batch = []
        self.current_batch_start = end_q_id + 1
        
        return str(filepath)
    
    def get_existing_files(self) -> List[str]:
        """获取已存在的 response 文件列表"""
        return sorted(glob.glob(str(self.output_dir / "responses_*.jsonl")))


# ==================== WandB 日志器 ====================

class WandBLogger:
    """WandB 监控日志器"""
    
    def __init__(
        self,
        enabled: bool = False,
        project: str = "rollout",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        run_name: Optional[str] = None,
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if self.enabled:
            if not WANDB_AVAILABLE:
                print("警告: wandb 未安装，禁用监控")
                self.enabled = False
                return
            
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=run_name,
                resume="allow",
            )
            print(f"✓ WandB 监控已启用: {wandb.run.url}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """记录指标"""
        if self.enabled and self.run:
            wandb.log(data, step=step)
    
    def log_batch_complete(
        self,
        batch_idx: int,
        questions_completed: int,
        total_questions: int,
        elapsed_time: float,
        prompt_tokens: int,
        completion_tokens: int,
        tokens_per_second: float,
    ):
        """记录批次完成信息"""
        if not self.enabled:
            return
        
        self.log({
            "batch/idx": batch_idx,
            "progress/questions_completed": questions_completed,
            "progress/completion_rate": questions_completed / total_questions,
            "time/elapsed_seconds": elapsed_time,
            "time/questions_per_second": questions_completed / elapsed_time if elapsed_time > 0 else 0,
            "tokens/prompt_total": prompt_tokens,
            "tokens/completion_total": completion_tokens,
            "tokens/per_second": tokens_per_second,
        }, step=questions_completed)
    
    def log_finish_reasons(self, finish_reason_counts: Dict[str, int]):
        """记录停止原因分布"""
        if not self.enabled:
            return
        
        for reason, count in finish_reason_counts.items():
            self.log({f"finish_reason/{reason}": count})
    
    def finish(self, status: str = "completed"):
        """结束运行"""
        if self.enabled and self.run:
            wandb.finish()


# ==================== Rollout 管理器 ====================

class RolloutManager:
    """
    Rollout 管理器
    
    整合 checkpoint、saver、logger 的高级接口
    """
    
    def __init__(
        self,
        output_base_dir: str,
        save_batch_size: int = 1000,
        enable_resume: bool = True,
        enable_wandb: bool = False,
        wandb_project: str = "rollout",
        wandb_entity: Optional[str] = None,
    ):
        self.output_base_dir = Path(output_base_dir)
        self.save_batch_size = save_batch_size
        self.enable_resume = enable_resume
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # 组件（延迟初始化）
        self.checkpoint_mgr: Optional[CheckpointManager] = None
        self.saver: Optional[ResponseSaver] = None
        self.logger: Optional[WandBLogger] = None
        self.meta: Optional[RolloutMeta] = None
        self.checkpoint: Optional[Checkpoint] = None
        
        # 运行时状态
        self.output_dir: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.is_resumed: bool = False
    
    def initialize(
        self,
        model_config: ModelConfig,
        sampling_params: SamplingParams,
        total_questions: int,
        formatted_prompts_path: str,
        sampled_questions_path: str,
    ) -> int:
        """
        初始化 rollout 管理器
        
        Returns:
            起始问题索引（用于续传）
        """
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否需要续传
        resume_dir = None
        if self.enable_resume:
            resume_dir = CheckpointManager.find_latest_incomplete_rollout(
                str(self.output_base_dir)
            )
        
        if resume_dir:
            # 续传模式
            self.output_dir = Path(resume_dir)
            self.is_resumed = True
            print(f"发现未完成的 rollout，将从断点续传: {resume_dir}")
        else:
            # 新建 rollout
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rollout_id = f"rollout_{timestamp}"
            self.output_dir = self.output_base_dir / rollout_id
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"创建新的 rollout: {self.output_dir}")
        
        # 初始化组件
        self.checkpoint_mgr = CheckpointManager(str(self.output_dir))
        self.saver = ResponseSaver(str(self.output_dir), self.save_batch_size)
        
        # 加载或创建元信息
        if self.is_resumed:
            self.meta = self.checkpoint_mgr.load_meta()
            self.checkpoint = self.checkpoint_mgr.load_checkpoint()
            start_index = self.checkpoint.last_completed_q_id + 1 if self.checkpoint else 0
            print(f"从问题 {start_index} 继续生成")
        else:
            # 创建新的元信息
            self.meta = RolloutMeta(
                rollout_id=self.output_dir.name,
                created_at=datetime.now().isoformat(),
                model_config=asdict(model_config),
                sampling_params=asdict(sampling_params),
                total_questions=total_questions,
                n_samples_per_question=sampling_params.n_samples,
                total_responses=total_questions * sampling_params.n_samples,
                formatted_prompts_path=formatted_prompts_path,
                sampled_questions_path=sampled_questions_path,
                save_batch_size=self.save_batch_size,
                output_dir=str(self.output_dir),
            )
            self.checkpoint_mgr.save_meta(self.meta)
            
            # 创建初始 checkpoint
            self.checkpoint = Checkpoint(
                rollout_id=self.output_dir.name,
                last_completed_q_id=-1,
                last_saved_batch_end=-1,
                updated_at=datetime.now().isoformat(),
            )
            self.checkpoint_mgr.save_checkpoint(self.checkpoint)
            start_index = 0
        
        # 初始化 WandB
        if self.enable_wandb:
            wandb_config = {
                **asdict(model_config),
                **asdict(sampling_params),
                "total_questions": total_questions,
                "save_batch_size": self.save_batch_size,
            }
            self.logger = WandBLogger(
                enabled=True,
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=wandb_config,
                run_name=self.output_dir.name,
            )
        else:
            self.logger = WandBLogger(enabled=False)
        
        self.start_time = time.time()
        
        return start_index
    
    def add_responses(
        self,
        q_id: int,
        question: str,
        gt_answer: str,
        prompt: str,
        responses: List[str],
        prompt_tokens: int,
        completion_tokens_list: List[int],
        finish_reasons: List[str],
    ):
        """添加一个问题的所有 responses"""
        # 保存 responses
        saved_file = self.saver.add_responses(
            q_id=q_id,
            question=question,
            gt_answer=gt_answer,
            prompt=prompt,
            responses=responses,
            prompt_tokens=prompt_tokens,
            completion_tokens_list=completion_tokens_list,
            finish_reasons=finish_reasons,
        )
        
        # 更新统计
        total_completion_tokens = sum(completion_tokens_list)
        self.checkpoint.total_prompt_tokens += prompt_tokens
        self.checkpoint.total_completion_tokens += total_completion_tokens
        
        for reason in finish_reasons:
            self.checkpoint.finish_reason_counts[reason] = \
                self.checkpoint.finish_reason_counts.get(reason, 0) + 1
        
        # 更新 checkpoint
        self.checkpoint.last_completed_q_id = q_id
        if saved_file:
            self.checkpoint.last_saved_batch_end = q_id
        
        # 定期保存 checkpoint
        if q_id % 100 == 0 or saved_file:
            self.checkpoint_mgr.save_checkpoint(self.checkpoint)
            
            # 更新元信息
            self.meta.completed_questions = q_id + 1
            self.meta.elapsed_time_seconds = time.time() - self.start_time
            self.meta.total_prompt_tokens = self.checkpoint.total_prompt_tokens
            self.meta.total_completion_tokens = self.checkpoint.total_completion_tokens
            self.meta.finish_reason_counts = self.checkpoint.finish_reason_counts.copy()
            self.checkpoint_mgr.save_meta(self.meta)
        
        # 记录到 WandB
        if self.logger and (q_id + 1) % 10 == 0:
            elapsed = time.time() - self.start_time
            tokens_total = self.checkpoint.total_prompt_tokens + self.checkpoint.total_completion_tokens
            self.logger.log_batch_complete(
                batch_idx=q_id // self.save_batch_size,
                questions_completed=q_id + 1,
                total_questions=self.meta.total_questions,
                elapsed_time=elapsed,
                prompt_tokens=self.checkpoint.total_prompt_tokens,
                completion_tokens=self.checkpoint.total_completion_tokens,
                tokens_per_second=tokens_total / elapsed if elapsed > 0 else 0,
            )
    
    def finalize(self, status: str = "completed"):
        """完成 rollout"""
        # 写入剩余的 responses
        self.saver.flush()
        
        # 更新最终状态
        self.meta.status = status
        self.meta.completed_questions = self.checkpoint.last_completed_q_id + 1
        self.meta.elapsed_time_seconds = time.time() - self.start_time
        self.meta.total_prompt_tokens = self.checkpoint.total_prompt_tokens
        self.meta.total_completion_tokens = self.checkpoint.total_completion_tokens
        self.meta.finish_reason_counts = self.checkpoint.finish_reason_counts.copy()
        
        self.checkpoint_mgr.save_meta(self.meta)
        self.checkpoint_mgr.save_checkpoint(self.checkpoint)
        
        # 如果 rollout 完成，合并所有分批文件
        if status == "completed":
            self._merge_response_files()
        
        # 记录最终统计
        if self.logger:
            self.logger.log_finish_reasons(self.meta.finish_reason_counts)
            self.logger.finish(status)
        
        # 打印统计
        self._print_summary()
    
    def _merge_response_files(self):
        """合并所有分批保存的 response 文件到一个完整的 JSONL 文件"""
        response_files = self.saver.get_existing_files()
        if not response_files:
            print("没有找到需要合并的 response 文件")
            return
        
        # 生成合并后的文件名
        merged_filename = f"all_responses_{self.meta.rollout_id}.jsonl"
        merged_filepath = self.output_dir / merged_filename
        
        print(f"\n合并 {len(response_files)} 个分批文件到: {merged_filename}")
        
        total_lines = 0
        with open(merged_filepath, 'w', encoding='utf-8') as f_out:
            for response_file in sorted(response_files):
                with open(response_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        f_out.write(line)
                        total_lines += 1
        
        print(f"✓ 合并完成: {total_lines} 条记录")
        print(f"  - 合并文件: {merged_filepath}")
        print(f"  - 原始分批文件已保留")
        
        # 更新 meta 信息
        self.meta.merged_file = str(merged_filepath)
        self.checkpoint_mgr.save_meta(self.meta)
    
    def _print_summary(self):
        """打印运行摘要"""
        print("\n" + "=" * 80)
        print("Rollout 完成摘要")
        print("=" * 80)
        print(f"Rollout ID: {self.meta.rollout_id}")
        print(f"状态: {self.meta.status}")
        print(f"输出目录: {self.output_dir}")
        print(f"\n数据统计:")
        print(f"  - 完成问题数: {self.meta.completed_questions} / {self.meta.total_questions}")
        print(f"  - 生成 responses: {self.meta.completed_questions * self.meta.n_samples_per_question}")
        print(f"\nToken 统计:")
        print(f"  - Prompt tokens: {self.meta.total_prompt_tokens:,}")
        print(f"  - Completion tokens: {self.meta.total_completion_tokens:,}")
        print(f"  - 总计: {self.meta.total_prompt_tokens + self.meta.total_completion_tokens:,}")
        print(f"\n停止原因分布:")
        for reason, count in self.meta.finish_reason_counts.items():
            print(f"  - {reason}: {count}")
        print(f"\n耗时: {self.meta.elapsed_time_seconds:.2f} 秒")
        if self.meta.elapsed_time_seconds > 0:
            qps = self.meta.completed_questions / self.meta.elapsed_time_seconds
            tps = (self.meta.total_prompt_tokens + self.meta.total_completion_tokens) / self.meta.elapsed_time_seconds
            print(f"  - 问题/秒: {qps:.2f}")
            print(f"  - Tokens/秒: {tps:.2f}")
        print("=" * 80)
    
    def get_output_dir(self) -> str:
        """获取输出目录路径"""
        return str(self.output_dir)

