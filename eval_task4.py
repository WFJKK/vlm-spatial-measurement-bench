import json, re, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = "You are inspecting a technical drawing with two holes. Use the scale bar to determine the vertical offset between the centers of H1 and H2. Respond with ONLY the offset in mm as a number, nothing else."
QUESTION = "What is the vertical offset between H1 and H2 in mm?"

def parse(t):
    try: return float(t.strip())
    except: pass
    m = re.search(r"[0-9]+\.?[0-9]*", t)
    return float(m.group(0)) if m else None

def infer(model, processor, path):
    image = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT + chr(10) + chr(10) + QUESTION}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"":0}, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

with open("dataset_task4/test/metadata.jsonl") as f:
    samples = [json.loads(l) for l in f]

maes = []
for i, s in enumerate(samples):
    resp = infer(model, processor, "dataset_task4/test/image_" + str(s["idx"]).zfill(4) + ".png")
    pred = parse(resp)
    gt = s["offset_mm"]
    if pred is not None:
        maes.append(abs(pred - gt))
    if (i+1) % 50 == 0:
        print("  [" + str(i+1) + "/200] MAE=" + str(round(np.mean(maes),2)) + " (parsed " + str(len(maes)) + "/" + str(i+1) + ")")

print("Task 4 Baseline MAE: " + str(round(np.mean(maes),2)))
print("Median AE: " + str(round(np.median(maes),2)))
print("Mean guess MAE: 1.85")
print("Parse rate: " + str(len(maes)) + "/" + str(len(samples)))

with open("dataset_task4/test_matched/metadata.jsonl") as fh:
    pairs = [json.loads(l) for l in fh]
for pm in pairs:
    pid = pm["pair_id"]
    pm["pred_a"] = parse(infer(model, processor, "dataset_task4/test_matched/pair_" + str(pid).zfill(3) + "_a.png"))
    pm["pred_b"] = parse(infer(model, processor, "dataset_task4/test_matched/pair_" + str(pid).zfill(3) + "_b.png"))
valid = [r for r in pairs if r.get("pred_a") is not None and r.get("pred_b") is not None]
gt_d = [r["offset_a"]-r["offset_b"] for r in valid]
pr_d = [r["pred_a"]-r["pred_b"] for r in valid]
corr = np.corrcoef(gt_d, pr_d)[0,1] if len(valid) > 2 else 0
same = sum(1 for r in valid if r["pred_a"]==r["pred_b"])/len(valid) if valid else 0
print("Matched pairs: " + str(len(valid)) + " valid")
print("Correlation: " + str(round(corr,4)))
print("Same answer: " + str(round(same*100,1)) + "%")
