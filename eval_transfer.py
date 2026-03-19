import json, re, numpy as np, torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CKPT = "checkpoints_task1_sft3/final"
PROMPT = "You are measuring a technical drawing. Use the scale bar to determine the distance between the two points. Respond with ONLY the distance in mm as a number, nothing else.\n\nWhat is the distance between P1 and P2 in mm?"

def parse(t):
    try: return float(t.strip())
    except: pass
    m = re.search(r"(\d+\.?\d*)", t)
    return float(m.group(1)) if m else None

def infer(model, processor, path):
    image = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("Loading model + SFT adapter...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, CKPT)
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

with open("dataset_task2/test/metadata.jsonl") as f:
    samples = [json.loads(l) for l in f]

maes = []
for i, s in enumerate(samples):
    resp = infer(model, processor, f"dataset_task2/test/image_{s['idx']:04d}.png")
    pred = parse(resp)
    if pred: maes.append(abs(pred - s["distance_mm"]))
    if (i+1) % 50 == 0: print(f"  [{i+1}/200] MAE={np.mean(maes):.2f}mm")

print(f"\nTask 1->2 TRANSFER MAE: {np.mean(maes):.2f}mm")
print(f"Task 2 baseline MAE:    10.92mm")
print(f"Task 1 SFT 3-epoch MAE: 1.62mm")
print(f"Task 1 baseline MAE:    7.72mm")

with open("dataset_task2/test_matched/metadata.jsonl") as fh:
    pairs = [json.loads(l) for l in fh]
for pm in pairs:
    pid = pm["pair_id"]
    pm["pred_a"] = parse(infer(model, processor, f"dataset_task2/test_matched/pair_{pid:03d}_a.png"))
    pm["pred_b"] = parse(infer(model, processor, f"dataset_task2/test_matched/pair_{pid:03d}_b.png"))
valid = [r for r in pairs if r.get("pred_a") and r.get("pred_b")]
gt_d = [r["dist_a"]-r["dist_b"] for r in valid]
pr_d = [r["pred_a"]-r["pred_b"] for r in valid]
corr = np.corrcoef(gt_d, pr_d)[0,1]
same = sum(1 for r in valid if r["pred_a"]==r["pred_b"])/len(valid)
print(f"Matched pair correlation: {corr:.4f}")
print(f"Same answer: {same:.1%}")
