# S-GRPO: Sampling-based GRPO with Decaying Reward
# Key difference from GRPO: generates 1 complete response, truncates m times, uses m+1 sequences as a group
export RAY_ADDRESS="100.121.104.230:53769"
set -x

python3 -m sgrpo.main_sgrpo \
    algorithm.adv_estimator=sgrpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files="/data/chenhaotian/verl/data/train.parquet" \
    data.val_files="/data/chenhaotian/verl/data/test.parquet" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/data/models/Qwen/Qwen3-8B" \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=1536 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    sgrpo.num_truncations=4 \
    'sgrpo.force_answer_prompt="Time is limited, stop thinking and start answering.\n</think>\n\n"' \
    sgrpo.answer_max_tokens=256 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_sgrpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_sgrpo' \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.rollout_data_dir="./outputs/sgrpo_samples" \
    trainer.rollout_data_dump_freq=20 $@
