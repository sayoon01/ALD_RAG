import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# 1) 임베딩 모델 설정 (검색용)
# ==============================

EMBED_MODEL_NAME = "thenlper/gte-small"
print(f"[+] Embedding model 로딩 중: {EMBED_MODEL_NAME}")
emb_model = SentenceTransformer(EMBED_MODEL_NAME)

# ==============================
# 2) 반도체 노트 (지식베이스 문장)
# ==============================

docs = [
    "ALD 공정은 Precursor A와 B를 번갈아 주입해 원자층 단위로 박막을 쌓는 공정이다.",
    "ALD는 Surface self-limiting 반응을 이용해 한 사이클당 일정 두께만큼만 증착된다.",
    "Precursor는 박막을 만들기 위한 원재료 기체이다.",
    "Precursor A는 웨이퍼 표면에 한 층만 흡착되면 반응이 멈춘다.",
    "Purge는 챔버 내부의 남아있는 기체를 제거하는 정리 단계이다.",
    "Purge는 A와 B가 공중에서 섞여 이상 반응을 일으키지 않도록 한다.",
    "플라즈마 자기장ALD는 라디칼을 이용해 낮은 온도에서도 높은 반응성을 확보한다.",
    "플라즈마를 사용하면 박막 밀도와 특성이 향상된다.",
    "MFC는 가스 유량을 정밀하게 제어하는 장치이다.",
    "유량은 sccm 단위로 표시되며 가스가 흐르는 양을 의미한다.",
    "APC 밸브는 챔버 압력을 일정하게 유지하기 위해 열리고 닫힌다.",
    "챔버는 반도체 공정이 실제로 일어나는 밀폐된 공간이다.",
    "VG12는 공정 챔버의 실제 압력을 측정하는 진공 게이지이다.",
    "VG13은 로드락 또는 전단부 압력을 측정하는 센서이다.",
    "로드락 챔버는 웨이퍼를 외부에서 내부로 이동시키기 위한 진공 전환 공간이다.",
    "압력 Overshoot는 가스가 과도하게 주입되거나 APC 제어가 늦어 발생한다.",
    "유량 Set 대비 실제 유량 오차가 크면 MFC 이상을 의심할 수 있다.",
    "Valve stuck-open 패턴은 밸브가 닫히지 않고 계속 열린 상태로 고착된 상황이다.",
    "반도체 웨이퍼는 여러 공정을 반복해 회로를 형성하는 얇은 실리콘 판이다.",
    "ALD 공정 조건은 시간 제어보다 표면 반응 특성에 의해 결정된다."
]

print(f"[+] 문서 개수: {len(docs)}")

# 문서 임베딩 계산
doc_embs = emb_model.encode(
    docs,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")
print("[+] 문서 임베딩 shape:", doc_embs.shape)

# ==============================
# 3) LLaMA LLM 설정
# ==============================

LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"[+] LLaMA 로딩 중: {LLM_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device.type != "cpu" else None
)

print(f"[+] LLaMA 로딩 완료 (device={device})")


# ==============================
# 4) 검색 함수 (Retrieval)
# ==============================

def retrieve(query: str, top_k: int = 3):
    q_emb = emb_model.encode([query], normalize_embeddings=True).astype("float32")[0]
    scores = np.dot(doc_embs, q_emb)
    idx_sorted = np.argsort(-scores)[:top_k]
    return [(docs[i], float(scores[i])) for i in idx_sorted]


# ==============================
# 5) LLaMA 기반 RAG 답변 생성 (개선된 버전)
# ==============================

def generate_answer(query: str, top_k: int = 3, max_new_tokens: int = 256):
    # 1) 관련 문장 검색
    retrieved = retrieve(query, top_k=top_k)

    # 검색 결과 기반 유사도 점수
    scores = [score for _, score in retrieved] if retrieved else []
    max_score = max(scores) if scores else 0.0

    # 2) 유사도가 너무 낮으면 hallucination 방지 → LLM 호출 안 함
    if not retrieved or max_score < 0.70:
        safe_msg = (
            "해당 노트에 없는 내용입니다.\n\n"
            "현재 보유한 노트에서 이 질문을 설명하기 위한 충분한 관련 문장을 찾지 못했어.\n"
            "노트에 없는 내용을 추측하지 않도록 답변 생성을 막았어.\n"
            "필요하면 관련 내용을 노트에 추가한 뒤 다시 질문해줘."
        )
        return safe_msg, retrieved

    # ===== LLM 호출 로직 (최대한 기존 유지) =====
    context_lines = [f"- {text}" for text, _ in retrieved]
    context_block = "\n".join(context_lines)

    system_prompt = (
        "너는 반도체 ALD, 플라즈마, 유량, 압력, 챔버 개념을 설명하는 조수야.\n"
        "반드시 한국어로 답하고, 영어 문장은 사용하지 마.\n"
        "아래 제공된 컨텍스트(노트) 내용만 근거로 설명해야 한다.\n"
        "컨텍스트에 관련 정보가 없으면 '해당 노트에 없는 내용입니다.'라고 답해야 한다.\n"
        "일반적인 공정 지식이나 추가 추론을 덧붙이지 마."
    )

    user_prompt = f"""
다음은 반도체 공정 관련 노트야.

[컨텍스트]
{context_block}

[질문]
{query}

위 컨텍스트만 활용해서 정확하게 설명해줘.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]  # 답변 부분만 추출
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return answer.strip(), retrieved


# ==============================
# 6) 인터랙티브 루프
# ==============================

if __name__ == "__main__":
    print("\n==============================")
    print("반도체 RAG 챗봇 (임베딩 + LLaMA)")
    print("질문을 입력하면 관련 노트 기반으로 답변해줘.")
    print("엔터만 누르면 종료.")
    print("==============================")

    while True:
        query = input("\n질문 입력: ").strip()
        if not query:
            print("끝냄!")
            break

        print("\n[검색 중...]")
        answer, retrieved_docs = generate_answer(query, top_k=3)

        print("\n[모델 답변]")
        print(answer)

        print("\n[참고한 컨텍스트 문장들]")
        for text, score in retrieved_docs:
            print(f"- score={score:.3f} | {text}")
