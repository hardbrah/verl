# PRM-GRPO Training Script for Qwen3-4B-Instruct-2507
# Uses PRM to score truncated thinking chains via latent states from layer 15
export RAY_ADDRESS="100.121.104.230:53769"
set -x

# Create log directory and log file with timestamp
LOG_DIR="/data/chenhaotian/verl/recipe/prm_grpo/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/prm_grpo_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

python3 -m recipe.prm_grpo.main_prm_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="/data/chenhaotian/verl/data/train.parquet" \
    data.val_files="/data/chenhaotian/verl/data/test.parquet" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="/data/models/Qwen/Qwen3-4B-Instruct-2507" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_prm_grpo' \
    trainer.experiment_name='prm_grpo_qwen3_4b_gsm8k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.rollout_data_dir="${LOG_DIR}/rollout_data" \
    trainer.validation_data_dir="${LOG_DIR}/validation_data" \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.prm.checkpoint_path="/data/chenhaotian/verl/checkpoints/latentqa/math_shepherd_20260116/best_model.pt" \
    reward_model.prm.backbone_path="/data/models/Qwen/Qwen3-4B-Instruct-2507" \
    reward_model.prm.extract_layer=15 \
    reward_model.prm.max_length=576 \
    $@ 2>&1 | tee ${LOG_FILE}
