# Qwen2.5-3B-Instruct 수학 문제풀이 특화 모델 (Math CoT SFT)

`Qwen/Qwen2.5-3B-Instruct` 모델을 수학 문제풀이(Chain-of-Thought)에 특화되도록 SFT(지도 미세조정)한 프로젝트입니다.
`<thought> ... </thought>` 태그로 풀이 과정을 서술하고, `\boxed{answer}` 형식으로 최종 정답을 출력하도록 학습되었습니다.

모델의 구글 드라이브 링크("Qwen_Math_LoRA"): [Qwen_Math_LoRA](https://drive.google.com/drive/folders/10Mqbg9JCbpAUrlIk2AkpUAN2AT3Pt5Rc?usp=sharing)
- 구조
Qwen_Math_LoRA
├── checkpoint-300
├── checkpoint-354
├── merged_vllm_model       # 학습된 모델
└── README.md


## 1. 모델 개요

| 항목 | 내용 |
|---|---|
| Base Model | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| 학습 방법 | QLoRA (4bit) SFT, [Unsloth](https://github.com/unslothai/unsloth) 활용 |
| Task | 수학 문제 풀이 (Chain-of-Thought 추론 + 정답 추출) |
| 출력 포맷 | `<thought>풀이 과정</thought>` + `The final answer is \boxed{answer}.` |
| 추론 방식 | vLLM 기반 다수결(Self-consistency, n=32) 샘플링 |

## 2. 파이프라인 개요

```
1) CoT 데이터 생성      : Qwen2.5-Math-1.5B-Instruct + vLLM으로 정답 근거(풀이과정) 생성
2) 데이터셋 병합         : (1)의 데이터 + GSM8K + MATH 데이터셋 병합
3) 토큰 수 점검          : 학습 전 데이터셋 토큰 길이 통계 확인
4) QLoRA SFT 학습        : Unsloth로 Qwen2.5-3B-Instruct 4bit QLoRA 학습
5) 모델 병합 및 저장      : LoRA 어댑터를 병합해 vLLM 서빙용 16bit 모델로 저장
6) 추론 및 제출 생성      : vLLM + 다수결 투표로 최종 제출 파일 생성
```

전체 과정은 `baseline.ipynb` 노트북 하나에 순서대로 정리되어 있으며, Google Colab 환경(GPU) 기준으로 작성되었습니다.

## 3. 학습 데이터

| 출처 | 설명 | 샘플 수 |
|---|---|---|
| 주최측 데이터 (`deep_chal_math_train.csv`) | Qwen2.5-Math-1.5B-Instruct로 생성한 CoT 풀이 중, 정답과 일치하는 것만 필터링 | 14,021개 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | `train` split에서 랜덤 샘플링 (seed=42) | 1,500개 |
| [MATH (EleutherAI/hendrycks_math)](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | 7개 서브셋(algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus) 병합 후 샘플링 | 7,500개 |
| **합계** | | **약 23,021개** |

> 주최측 원본 데이터(`deep_chal_math_train.csv`, `train_filtered_ids.csv`)는 대회/과제 제공 파일로 본 저장소에는 포함되어 있지 않습니다. 재현 시 별도로 준비해 루트 경로에 위치시켜야 합니다.

## 4. 환경 및 요구 사항

- Google Colab (GPU 런타임)
  - 학습 데이터 생성 및 SFT 학습: **T4 GPU** (속도가 매우 느립니다. 가능하면 A100/L4 권장)
  - 최종 추론(제출 파일 생성): **A100 GPU**
- Python 3
- 주요 라이브러리
  ```bash
  pip install vllm pandas tqdm
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers "trl<0.0.27" trl peft accelerate bitsandbytes
  pip install transformers datasets
  ```
  > 노트북 내에서는 vLLM 설치 전 기존 `torch`/`torchvision`/`torchaudio`/`vllm`을 완전히 삭제한 뒤 재설치합니다(버전 충돌 방지).

> **왜 vLLM 실행 코드를 `.py` 스크립트로 분리했나요?**
> Jupyter/Colab의 `ipykernel`은 출력을 가로채는 가짜 stdout을 쓰는데, 여기엔 OS가 부여하는 진짜 파일 디스크립터(`fileno`)가 없습니다. vLLM은 이 `fileno`를 요구하다 에러를 내며, CUDA 멀티프로세싱(spawn) 환경에서는 노트북 셀에 임시로 적용한 패치도 자식 프로세스로 전달되지 않아 문제가 재발합니다. `%%writefile`로 스크립트를 만들어 `!python`으로 실행하면 진짜 터미널 stdout에서 동작하므로 이 문제가 원천적으로 해결됩니다.

## 5. 재현 방법

### Step 1. CoT 학습 데이터 생성
`Qwen/Qwen2.5-Math-1.5B-Instruct` 모델과 vLLM을 이용해 주최측 문제에 대한 단계별 풀이(CoT)를 생성합니다.
- 입력: `deep_chal_math_train.csv`, `train_filtered_ids.csv`(제외할 id 목록)
- 정답과 일치하는 풀이만 채택(정답이 다르면 재시도/스킵)
- 결과: `generated_comp_dataset.csv` (14,021개 확보까지 반복 실행)

```bash
python create_cot_data.py
```

### Step 2. 데이터셋 병합
Step 1 결과 + GSM8K + MATH 데이터셋을 동일한 `messages` 포맷(system/user/assistant)으로 통일해 병합합니다.
- System prompt: `<thought>` 태그로 풀이, `\boxed{answer}`로 정답 표기하도록 지시
- 결과: `final_math_train_v3_dataset` (HuggingFace `datasets` 저장 형식)

### Step 3. 토큰 수 점검 (선택)
`Qwen/Qwen2.5-3B` 토크나이저 기준으로 샘플별 토큰 수, 평균/최대 토큰 수, `max_seq_length`(2048) 초과 비율을 확인합니다.

### Step 4. QLoRA SFT 학습
Unsloth `FastLanguageModel`로 4bit 양자화된 `Qwen/Qwen2.5-3B-Instruct`를 로드하고 LoRA 어댑터를 붙여 학습합니다.

**LoRA 설정**

| 파라미터 | 값 |
|---|---|
| r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| bias | none |
| gradient_checkpointing | unsloth |

**학습 하이퍼파라미터 (SFTConfig)**

| 파라미터 | 값 |
|---|---|
| max_length | 2048 |
| packing | True |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 8 (effective batch size 16) |
| num_train_epochs | 1 |
| warmup_ratio | 0.05 |
| learning_rate | 3e-5 |
| optimizer | adamw_8bit |
| precision | bf16 (미지원 시 fp16) |
| save_steps / save_total_limit | 100 / 2 |

- Google Drive를 마운트해 체크포인트를 백업하며, 중간 저장은 로컬(Colab) 디스크에 먼저 수행 후 Drive로 복사합니다(I/O 병목 방지).
- 이전 체크포인트가 있으면 자동으로 이어서 학습(`resume_from_checkpoint`)합니다.

### Step 5. 모델 병합 및 저장
학습된 LoRA 어댑터를 base 모델에 병합(`save_pretrained_merged`, `merged_16bit`)하여 vLLM에서 바로 로드 가능한 형태로 저장합니다.
- 저장 경로: `Qwen_Math_LoRA/merged_vllm_model` (Google Drive)

### Step 6. 추론 및 제출 파일 생성
병합된 모델을 vLLM으로 로드하고, 문제당 32회 샘플링 후 다수결(최빈값)로 최종 정답을 결정합니다.

**추론 설정**

| 파라미터 | 값 |
|---|---|
| max_model_len | 2048 |
| gpu_memory_utilization | 0.92 |
| num_votes (n) | 32 |
| temperature | 0.4 |
| top_p | 0.9 |
| max_tokens | 2048 |

```bash
python create_submission.py
```
- 입력: `deep_chal_math_leaderboard_filtered.csv`
- 출력: `submission.csv` (`id`, `answer` 컬럼)

> ⚠️ 런타임을 새로 연결한 직후 실행하는 것을 권장합니다(이전 세션의 vLLM/torch 프로세스 잔여물로 인한 충돌 방지).

## 6. 저장소 구조

```
.
├── baseline.ipynb              # 전체 파이프라인 노트북 (데이터 생성 ~ 추론)
├── final_math_train_v3_dataset # 최종 학습 데이터 셋 폴더
└── README.md                   # 학습된 모델은 README.md 파일 상단 구글드라이브 링크에서 찾을 수 있습니다.
```

## 7. 주의 사항

- 대회/과제 제공 원본 데이터(`deep_chal_math_train.csv`, `train_filtered_ids.csv`, `deep_chal_math_leaderboard_filtered.csv`)는 본 저장소에 포함하지 않았습니다.
- T4 GPU에서는 학습 및 데이터 생성 속도가 매우 느립니다. 가능하면 A100/L4 등 상위 GPU 사용을 권장합니다.
- vLLM 설치 시 기존 torch 계열 패키지와 충돌이 발생할 수 있어, 노트북에서는 삭제 후 재설치하는 방식을 사용합니다.

## 8. 라이선스

Base 모델인 Qwen2.5-3B-Instruct의 라이선스를 따릅니다. 사용한 공개 데이터셋(GSM8K, MATH)은 각 데이터셋의 라이선스를 확인해 주세요.
