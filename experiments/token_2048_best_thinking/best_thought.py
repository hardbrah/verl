import json
from collections import defaultdict
from verl.utils.reward_score.math_dapo import compute_score

def select_best_thought():
    json_path = "/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/final_complete_rollouts_100_nofake.jsonl"
    results = []
    with open(json_path, "r") as f:
        data = [json.loads(line) for line in f]

    q_id2token_ids = defaultdict(list[int])
    token_id2acc = defaultdict(int)
    token_id2total = defaultdict(int)
    token_id2idx = defaultdict(int)
    for idx,item in enumerate(data):
        solution_str = item["complete_rollout"]
        ground_truth = item["gt_answer"]
        token_id = item["token_id"]
        q_id = item["q_id"]
        result = compute_score(solution_str, ground_truth)
        q_id2token_ids[q_id].append(token_id)
        token_id2acc[token_id] += result["acc"]
        token_id2total[token_id] += 1
        token_id2idx[token_id] = idx
    
    logs = []
    for q_id,token_ids in q_id2token_ids.items():
        best_token_id = max(token_ids, key=lambda x: token_id2acc[x])
        best_idx = token_id2idx[best_token_id]
        best_data = data[best_idx]
        best_data.update({"acc":token_id2acc[best_token_id]/token_id2total[best_token_id]})
        results.append(best_data)

        small_dict = {k:token_id2acc[k] for k in token_ids}
        sorted_pairs = sorted(small_dict.items(), key=lambda x: x[1], reverse=True)
        logs.append({
            "q_id":q_id,
            "best_token_id":best_token_id,
            "best_idx":best_idx,
            "counter":[{
                "token_id":k,
                "acc":v
            } for k,v in sorted_pairs]
        })

    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/best_thought.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    with open("/mnt/nas/chenhaotian/verl/experiments/token_2048_best_thinking/outputs/best_thought_logs.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"Selected {len(results)} best thoughts")

if __name__ == "__main__":
    select_best_thought()