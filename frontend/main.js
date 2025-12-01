// 백엔드 FastAPI 주소
// SSH 터널 쓰면 보통 http://localhost:8000/chat 로 접근
const API_URL = "http://localhost:8000/chat";

const chatWindow = document.getElementById("chat-window");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const contextList = document.getElementById("context-list");
const topKInput = document.getElementById("top-k-input");
const maxTokensInput = document.getElementById("max-tokens-input");

// 간단한 채팅 로그 메모리
const messages = []; // { role: "user" | "assistant", content: string, timestamp: Date }

function appendMessage(role, content) {
  const msg = {
    role,
    content,
    timestamp: new Date(),
  };
  messages.push(msg);
  renderMessages();
}

function renderMessages() {
  chatWindow.innerHTML = "";

  messages.forEach((msg) => {
    const item = document.createElement("div");
    item.classList.add("chat-message", msg.role === "user" ? "from-user" : "from-bot");

    const meta = document.createElement("div");
    meta.classList.add("message-meta");
    meta.textContent =
      msg.role === "user"
        ? "질문"
        : "모델 답변";

    const body = document.createElement("div");
    body.classList.add("message-body");
    body.textContent = msg.content;

    item.appendChild(meta);
    item.appendChild(body);
    chatWindow.appendChild(item);
  });

  // 항상 맨 아래로 스크롤
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function renderContexts(contexts) {
  contextList.innerHTML = "";

  if (!contexts || contexts.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "컨텍스트가 없습니다.";
    contextList.appendChild(emptyItem);
    return;
  }

  contexts.forEach((ctx) => {
    const li = document.createElement("li");
    const score = typeof ctx.score === "number" ? ctx.score.toFixed(3) : ctx.score;

    li.innerHTML = `
      <div class="ctx-score">score=${score}</div>
      <div class="ctx-text">${ctx.text}</div>
    `;

    contextList.appendChild(li);
  });
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    alert("질문을 입력해줘!");
    return;
  }

  const topK = parseInt(topKInput.value || "3", 10);
  const maxNewTokens = parseInt(maxTokensInput.value || "256", 10);

  // 사용자 메시지 UI에 추가
  appendMessage("user", question);
  questionInput.value = "";

  // 상태 표시
  statusText.textContent = "모델이 생각 중입니다... (GPU에서 RAG + LLaMA 실행 중)";
  statusText.classList.add("loading");

  // 이전 컨텍스트 초기화
  contextList.innerHTML = "";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: topK,
        max_new_tokens: maxNewTokens,
      }),
    });

    if (!res.ok) {
      statusText.textContent = `API 요청 실패 (status: ${res.status})`;
      statusText.classList.remove("loading");
      return;
    }

    const data = await res.json();

    // 모델 답변 추가
    appendMessage("assistant", data.answer || "(빈 답변)");

    // 컨텍스트 렌더링
    renderContexts(data.contexts || []);

    statusText.textContent = "";
    statusText.classList.remove("loading");
  } catch (err) {
    console.error(err);
    statusText.textContent = "요청 중 에러가 발생했습니다. 콘솔을 확인해 주세요.";
    statusText.classList.remove("loading");
  }
}

// 버튼 클릭 시 전송
sendBtn.addEventListener("click", sendQuestion);

// Ctrl+Enter / Cmd+Enter 로 전송
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendQuestion();
  }
});
