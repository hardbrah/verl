# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entry point for PRM-GRPO training.

This module extends the standard PPO training with PRM-based rewards.
It creates a custom TaskRunner that uses PRMRewardModelWorker instead of
the standard RewardModelWorker.
"""

import os
import sys

import hydra
import ray
from omegaconf import OmegaConf

# Add the recipe directory to path for imports
recipe_dir = os.path.dirname(os.path.abspath(__file__))
if recipe_dir not in sys.path:
    sys.path.insert(0, recipe_dir)

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available


_VERL_PLUGIN_REGISTERED = False


def _register_verl_config_searchpath():
    """
    Register verl's config directory to Hydra's search path.
    This allows prm_grpo.yaml to reference configs from verl/trainer/config/
    """
    global _VERL_PLUGIN_REGISTERED
    if _VERL_PLUGIN_REGISTERED:
        return
    
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    class VerlConfigSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):
            # Add verl's trainer config directory to the search path
            search_path.append(
                provider="verl-trainer-config",
                path="pkg://verl.trainer.config",
            )

    # Register the plugin
    plugins = Plugins.instance()
    plugins.register(VerlConfigSearchPathPlugin)
    _VERL_PLUGIN_REGISTERED = True


# Register the search path plugin before Hydra initialization
_register_verl_config_searchpath()


class PRMTaskRunner(TaskRunner):
    """
    Custom TaskRunner for PRM-GRPO.
    
    This extends the standard TaskRunner to use PRMRewardModelWorker
    instead of the standard RewardModelWorker for reward computation.
    """
    
    def add_reward_model_worker(self, config):
        """Add PRM reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role
        
        if config.reward_model.enable:
            # Import PRMRewardModelWorker from local module
            from prm_worker import PRMRewardModelWorker
            
            print(f"[PRMTaskRunner] Using PRMRewardModelWorker for reward computation")
            print(f"  - PRM checkpoint: {config.reward_model.prm.checkpoint_path}")
            print(f"  - PRM backbone: {config.reward_model.prm.backbone_path}")
            print(f"  - Extract layer: {config.reward_model.prm.get('extract_layer', 15)}")
            
            self.role_worker_mapping[Role.RewardModel] = ray.remote(PRMRewardModelWorker)
            
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"
                

def run_prm_grpo(config, task_runner_class=None) -> None:
    """
    Initialize Ray cluster and run distributed PRM-GRPO training process.
    
    Args:
        config: Training configuration object containing all necessary parameters.
        task_runner_class: Optional custom TaskRunner class. Defaults to PRMTaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        
        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars
            
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"[PRM-GRPO] Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
        
    # Use PRMTaskRunner by default
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(PRMTaskRunner)
        
    # Create and run the task runner
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available
        
        assert is_nvtx_available(), "nvtx is not available. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
        
    ray.get(runner.run.remote(config))
    
    # Optional timeline trace
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)
        print(f"[PRM-GRPO] Timeline saved to {timeline_json_file}")
        

# Use PRM-GRPO specific config from recipe directory
# The config_path points to the recipe's config folder which contains prm_grpo.yaml
# prm_grpo.yaml uses "defaults" to inherit from verl's standard configs while
# overriding reward_model with prm_reward_model.yaml
@hydra.main(config_path="config", config_name="prm_grpo", version_base=None)
def main(config):
    """Main entry point for PRM-GRPO training with Hydra configuration."""
    print("[PRM-GRPO] Starting PRM-GRPO training...")
    print(f"[PRM-GRPO] Reward model enabled: {config.reward_model.enable}")
    if config.reward_model.enable:
        print(f"[PRM-GRPO] PRM checkpoint: {config.reward_model.prm.checkpoint_path}")
        print(f"[PRM-GRPO] PRM backbone: {config.reward_model.prm.backbone_path}")
        print(f"[PRM-GRPO] PRM extract layer: {config.reward_model.prm.extract_layer}")
    run_prm_grpo(config)
    

if __name__ == "__main__":
    main()
