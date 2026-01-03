# เริ่มใหม่ตั้งแต่ต้น: Train LoRA (Space GPU) → Merge Full Model → Deploy Inference Endpoint

เอกสารนี้เป็น playbook แบบ “ทำตามได้ทีละขั้น” สำหรับเริ่มใหม่ทั้งหมดหลังจาก **ลบ model repo และ Space เดิมแล้ว**

> เป้าหมาย
>
> 1. เทรน LoRA ให้โทนเหมือนคุณ
> 2. merge เป็น full model
> 3. deploy เป็น Inference Endpoint เพื่อเรียกผ่าน API ได้แน่นอน

---

## A) เตรียมของก่อนเริ่ม

### A1) สร้าง Hugging Face Token ใหม่ (สำคัญ)

1. ไปที่ Settings → Access Tokens
2. สร้าง token แบบ **Write**
3. เก็บค่า token (ขึ้นต้น `hf_...`) ไว้ใช้เป็น secret เท่านั้น

> หมายเหตุ: ถ้าเคยแปะ token ในที่สาธารณะ ให้ Revoke ตัวเก่าให้หมดก่อน

### A2) เลือก base model (แนะนำเริ่ม)

* แนะนำเริ่มจาก **Qwen/Qwen3-0.6B-Base** เพื่อเทรนเร็ว/ถูก
* (ถ้าต้องการคุณภาพสูงขึ้นค่อยขยับเป็น 4B/7B ทีหลัง)

### A3) เตรียม dataset แบบ SFT สำหรับ “โทนคุณ”

สร้างไฟล์ `train.jsonl` รูปแบบ conversational:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

แนวทางทำข้อมูล:

* 100–500 ตัวอย่างคุณภาพสูงเริ่มได้
* ให้ assistant ตอบเป็น “สไตล์คุณ” (bullet, กระชับ, trade-off, action)
* หลีกเลี่ยงข้อมูลส่วนตัวอ่อนไหว

---

## B) สร้าง Repo 2 อันบน Hugging Face

### B1) Repo สำหรับ LoRA adapter (หลังเทรนจะ push มาที่นี่)

* ชื่อแนะนำ: `issarapong/issarapong-lora`
* Type: Model
* Public/Private ตามสะดวก (แนะนำ Private ถ้ามีข้อมูลเฉพาะ)

### B2) Repo สำหรับ Full merged model

* ชื่อแนะนำ: `issarapong/issarapong-tone`
* Type: Model
* Public/Private ตามสะดวก

> ทำไมต้องแยก 2 repo:
>
> * LoRA repo เล็ก/จัดเวอร์ชันง่าย
> * Full model repo พร้อม deploy ง่าย

---

## C) Train LoRA บน Hugging Face Space (GPU)

### C1) สร้าง Space สำหรับเทรน

* SDK: **Gradio**
* Template: **Blank**
* Hardware: **Nvidia T4 medium (8 vCPU / 30GB)**
* Dev Mode: OFF

### C2) ตั้ง Secrets ใน Space

ไปที่ Space → Settings → Secrets

* `HF_TOKEN` = token ใหม่ (Write)
* `LORA_REPO` = `issarapong/issarapong-lora`
* `BASE_MODEL` = `Qwen/Qwen3-0.6B-Base`

### C3) เพิ่มไฟล์ใน Space repo

ต้องมีอย่างน้อย 3 ไฟล์นี้:

1. `train.jsonl` (อัปโหลดไฟล์ dataset)
2. `requirements.txt`
3. `app.py`

#### requirements.txt (เทรน LoRA)

```txt
torch
transformers>=4.41.0
datasets
accelerate
peft
trl
safetensors
huggingface_hub
gradio>=4.0.0
```

#### app.py (Train + Push LoRA)

> กดปุ่มครั้งเดียว แล้วมันจะเทรนและ push adapter ไปที่ `LORA_REPO`

```python
import os, traceback
import gradio as gr
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch


def train_and_push():
    hf_token = os.environ.get("HF_TOKEN")
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B-Base")
    lora_repo = os.environ.get("LORA_REPO")

    if not hf_token:
        return "❌ Missing HF_TOKEN in Settings → Secrets (ต้องเป็น Write token)"
    if not lora_repo:
        return "❌ Missing LORA_REPO (เช่น issarapong/issarapong-lora)"

    try:
        login(token=hf_token)

        # ต้องมีไฟล์ train.jsonl ใน Space repo
        ds = load_dataset("json", data_files="train.jsonl", split="train")

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
            args=SFTConfig(
                output_dir="lora_out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=2e-4,
                num_train_epochs=2,
                logging_steps=10,
                save_steps=200,
            ),
        )

        trainer.train()

        # push เฉพาะ adapter
        trainer.model.push_to_hub(lora_repo, safe_serialization=True)
        tokenizer.push_to_hub(lora_repo)

        return (
            "✅ Train + Push complete
"
            f"Base: {base_model}
"
            f"LoRA repo: {lora_repo}
"
            "เช็ค repo ว่ามี adapter_config.json + adapter_model.safetensors"
        )

    except Exception:
        return "❌ Error:
" + traceback.format_exc()


with gr.Blocks() as demo:
    gr.Markdown("# Train LoRA Persona → Push to Hub")
    gr.Markdown("อัปโหลด train.jsonl แล้วกดปุ่ม Train + Push")
    btn = gr.Button("Train + Push")
    out = gr.Textbox(lines=25, label="Logs")
    btn.click(fn=train_and_push, inputs=None, outputs=out)

demo.launch()
```

### C4) เช็คว่าเทรนสำเร็จ

ไปที่ repo `issarapong/issarapong-lora` แล้วต้องเห็นไฟล์หลัก ๆ:

* `adapter_config.json`
* `adapter_model.safetensors` (หรือ `.bin`)
* (มี tokenizer/config บ้างก็ปกติ)

---

## D) Merge LoRA → Full Model (ทำบน Space GPU)

### D1) สร้าง Space ใหม่สำหรับ merge (หรือใช้ Space เดิมแล้วเปลี่ยนโค้ด)

* SDK: Gradio (Blank)
* Hardware: T4 medium

### D2) ตั้ง Secrets สำหรับ merge

* `HF_TOKEN` = token (Write)
* `BASE_MODEL` = `Qwen/Qwen3-0.6B-Base`
* `LORA_REPO` = `issarapong/issarapong-lora`
* `OUTPUT_REPO` = `issarapong/issarapong-tone`

### D3) requirements.txt (merge)

```txt
transformers>=4.41.0
peft>=0.11.0
accelerate>=0.31.0
huggingface_hub>=0.23.0
torch
safetensors
gradio>=4.0.0
```

### D4) app.py (Merge + Push Full Model)

```python
import os
import traceback
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login


def merge_and_push():
    hf_token = os.environ.get("HF_TOKEN")
    base_model = os.environ.get("BASE_MODEL")
    lora_repo = os.environ.get("LORA_REPO")
    output_repo = os.environ.get("OUTPUT_REPO")

    if not hf_token:
        return "❌ Missing HF_TOKEN"
    if not base_model:
        return "❌ Missing BASE_MODEL"
    if not lora_repo:
        return "❌ Missing LORA_REPO"
    if not output_repo:
        return "❌ Missing OUTPUT_REPO"

    try:
        login(token=hf_token)

        logs = []
        logs.append(f"Base: {base_model}")
        logs.append(f"LoRA: {lora_repo}")
        logs.append(f"Out : {output_repo}")

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(model, lora_repo)
        model = model.merge_and_unload()

        model.push_to_hub(output_repo, safe_serialization=True)
        tokenizer.push_to_hub(output_repo)

        logs.append("✅ Merge + Push done")
        return "
".join(logs)

    except Exception:
        return "❌ Error:
" + traceback.format_exc()


with gr.Blocks() as demo:
    gr.Markdown("# Merge LoRA → Full Model")
    btn = gr.Button("Merge + Push")
    out = gr.Textbox(lines=25, label="Logs")
    btn.click(fn=merge_and_push, inputs=None, outputs=out)

demo.launch()
```

### D5) เช็ค full model repo

ไปที่ `issarapong/issarapong-tone` แล้วต้องเห็นไฟล์โมเดลเต็ม เช่น:

* `config.json`
* `model.safetensors` (หรือหลาย shard)
* tokenizer files

---

## E) Deploy เป็น Inference Endpoint (เรียก API ได้แน่นอน)

### E1) สร้าง Endpoint

1. ไปที่ Inference Endpoints
2. Create Endpoint
3. เลือก model: `issarapong/issarapong-tone`
4. เลือก instance (เริ่มจาก GPU เล็ก ๆ พอ)
5. Deploy

### E2) เรียกใช้งาน API

Endpoint จะให้ URL + วิธีเรียก (มักจะเป็น OpenAI-compatible หรือ text-generation)
ใช้ token เดียวกัน (หรือ endpoint token ตามที่ระบบให้)

---

## F) Checklist สั้น ๆ (กันหลง)

* [ ] Token ใหม่ (Write)
* [ ] สร้าง 2 repo: `*-lora` และ `*-tone`
* [ ] Space Train: มี `train.jsonl`, `requirements.txt`, `app.py`
* [ ] หลังเทรน: `*-lora` ต้องมี `adapter_config.json` + `adapter_model.safetensors`
* [ ] Space Merge: push ไป `*-tone`
* [ ] Deploy Endpoint จาก `*-tone`

---

## G) หมายเหตุเรื่อง base model (0.6B vs 4B)

* **เทรน LoRA แนะนำเริ่มที่ base (0.6B)**: ถูก/เร็ว/ควบคุมง่าย
* ส่วน **4B-Instruct เหมาะเป็นโมเดลเรียกผ่าน Router** (แต่ไม่ใช่ทางลัดสำหรับ “มีโมเดลคุณเองผ่าน Router”)

---

## จุดที่คุณทำต่อทันที

1. สร้าง repo `issarapong/issarapong-lora` และ `issarapong/issarapong-tone`
2. ทำ `train.jsonl` เวอร์ชันแรก (อย่างน้อย 50–100 ตัวอย่าง)
3. สร้าง Space Train แล้วอัปโหลด 3 ไฟล์ (train.jsonl + requirements.txt + app.py)

พอคุณทำข้อ 1–3 เสร็จ แจ้งผมได้เลย ผมจะช่วยเช็คว่า dataset/เทรนเซ็ตอัปถูก และไล่ไปจน deploy endpoint สำเร็จ
