// ============================================
// 설정
// ============================================
const API_BASE = "http://127.0.0.1:8000";

// ============================================
// DOM 요소 참조
// ============================================
// DOM 요소 참조 (초기화는 DOMContentLoaded에서)
let els = {};

function initElements() {
  els = {
    // 상태
    status: document.getElementById("api-status"),
    statusText: document.querySelector("#api-status .status-text"),
    statusDot: document.querySelector("#api-status .status-dot"),
    
    // 시스템 정보
    systemInfo: document.getElementById("system-info"),
    infoDevice: document.getElementById("info-device"),
    infoDocs: document.getElementById("info-docs"),
    infoKeywords: document.getElementById("info-keywords"),
    
    // 질문 입력
    question: document.getElementById("question"),
    sendBtn: document.getElementById("send-btn"),
    clearBtn: document.getElementById("clear-answer-btn"),
    infoLine: document.getElementById("info-line"),
    
    // 옵션
    topkInput: document.getElementById("topk-input"),
    topkLabel: document.getElementById("topk-label"),
    maxTokensInput: document.getElementById("max-tokens-input"),
    maxTokensLabel: document.getElementById("max-tokens-label"),
    keywordSelect: document.getElementById("keyword-select"),
    contextOnly: document.getElementById("context-only"),
    debugFlag: document.getElementById("debug-flag"),
    
    // 결과
    answer: document.getElementById("answer"),
    contexts: document.getElementById("contexts"),
    contextCount: document.getElementById("context-count"),
    keywordStats: document.getElementById("keyword-stats"),
    refreshStatsBtn: document.getElementById("refresh-stats-btn"),
  };
}

// ============================================
// 상태 표시 유틸
// ============================================
function setStatusOk(text) {
  if (els.statusText) els.statusText.textContent = text;
  els.status.classList.remove("status-error");
  els.status.classList.add("status-ok");
}

function setStatusError(text) {
  if (els.statusText) els.statusText.textContent = text;
  els.status.classList.remove("status-ok");
  els.status.classList.add("status-error");
}

function setInfo(text, type = "info") {
  if (!els.infoLine) return;
  
  els.infoLine.textContent = text || "";
  els.infoLine.className = "info-line";
  
  if (type === "error") {
    els.infoLine.style.color = "var(--danger)";
  } else if (type === "success") {
    els.infoLine.style.color = "var(--success)";
  } else {
    els.infoLine.style.color = "var(--text-sub)";
  }
}

// ============================================
// 시스템 정보 표시 (MODEL_INFO 활용)
// ============================================
function renderSystemInfo(data) {
  if (!data || !els.systemInfo) return;
  
  // num_docs가 있으면 우선 사용, 없으면 keywords에서 계산
  const numDocs = data.num_docs !== undefined 
    ? data.num_docs 
    : Object.values(data.keywords || {}).reduce((sum, count) => sum + count, 0);
  
  // keyword_list가 있으면 우선 사용, 없으면 keywords에서 추출
  const keywordList = data.keyword_list && data.keyword_list.length > 0
    ? data.keyword_list
    : Object.keys(data.keywords || {}).filter(k => k && k !== "unknown");
  
  // 시스템 정보 표시
  if (els.infoDevice) {
    els.infoDevice.textContent = "GPU/CPU";
  }
  
  if (els.infoDocs) {
    els.infoDocs.textContent = `${numDocs}개`;
  }
  
  if (els.infoKeywords) {
    els.infoKeywords.textContent = keywordList.length > 0 
      ? `${keywordList.length}개 (${keywordList.slice(0, 3).join(", ")}${keywordList.length > 3 ? "..." : ""})`
      : "없음";
  }
  
  els.systemInfo.style.display = "flex";
}

// ============================================
// API 상태 및 키워드 로딩
// ============================================
async function loadApiStatusAndKeywords() {
  try {
    const res = await fetch(`${API_BASE}/`);
    if (!res.ok) {
      setStatusError(`API 응답 실패: ${res.status}`);
      setInfo(`서버 응답 오류: ${res.status}`, "error");
      return;
    }
    
    const data = await res.json();
    setStatusOk("API 연결 성공");
    setInfo("", "success");
    
    // 시스템 정보 표시
    renderSystemInfo(data);
    
    // 키워드 정보 처리
    if (data.keywords && Object.keys(data.keywords).length > 0) {
      renderKeywordStats(data.keywords);
      fillKeywordSelect(Object.keys(data.keywords));
    } else {
      // /keywords 엔드포인트로 재시도
      await loadKeywordsFallback();
    }
  } catch (err) {
    console.error("[loadApiStatusAndKeywords 오류]", err);
    setStatusError("API 연결 실패");
    setInfo(`연결 오류: ${err.message || "서버에 연결할 수 없습니다"}`, "error");
    
    // 키워드 통계 영역에 오류 표시
    if (els.keywordStats) {
      els.keywordStats.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <p>API 연결 실패<br/>서버가 실행 중인지 확인하세요</p>
        </div>
      `;
    }
  }
}

async function loadKeywordsFallback() {
  try {
    const res = await fetch(`${API_BASE}/keywords`);
    if (!res.ok) {
      console.warn("[keywords fallback] 응답 실패:", res.status);
      return;
    }
    
    const stats = await res.json();
    if (stats && !stats.error) {
      renderKeywordStats(stats);
      fillKeywordSelect(Object.keys(stats));
    } else if (stats.error) {
      console.error("[keywords] 오류:", stats.error);
    }
  } catch (err) {
    console.error("[keywords fallback] 실패", err);
  }
}

function fillKeywordSelect(keywordList) {
  if (!els.keywordSelect) return;
  
  els.keywordSelect.innerHTML = '<option value="">(전체 키워드)</option>';
  
  keywordList
    .filter((kw) => kw && kw !== "unknown")
    .sort()
    .forEach((kw) => {
      const opt = document.createElement("option");
      opt.value = kw;
      opt.textContent = kw;
      els.keywordSelect.appendChild(opt);
    });
}

function renderKeywordStats(statsObj) {
  if (!els.keywordStats) return;
  
  els.keywordStats.innerHTML = "";
  
  const entries = Object.entries(statsObj || {});
  if (!entries.length) {
    els.keywordStats.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📊</div>
        <p>키워드 통계가 없습니다.</p>
      </div>
    `;
    return;
  }
  
  const total = entries.reduce((sum, [, v]) => sum + v, 0);
  
  const table = document.createElement("table");
  table.className = "keyword-table";
  
  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>키워드</th>
      <th>문장 수</th>
      <th>비율(%)</th>
    </tr>
  `;
  table.appendChild(thead);
  
  const tbody = document.createElement("tbody");
  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([kw, count]) => {
      const tr = document.createElement("tr");
      const ratio = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
      tr.innerHTML = `
        <td>${kw}</td>
        <td>${count}</td>
        <td>${ratio}</td>
      `;
      tbody.appendChild(tr);
    });
  table.appendChild(tbody);
  
  els.keywordStats.appendChild(table);
}

// ============================================
// 슬라이더 라벨 동기화 (init 함수로 이동됨)
// ============================================

// ============================================
// 질문 전송 및 답변 처리
// ============================================
async function sendQuestion() {
  const question = els.question?.value.trim() || "";
  const top_k = parseInt(els.topkInput?.value, 10) || 3;
  const max_new_tokens = parseInt(els.maxTokensInput?.value, 10) || 256;
  const filter_keyword = els.keywordSelect?.value || null;
  const context_only = els.contextOnly?.checked || false;
  const debug = els.debugFlag?.checked || false;
  
  if (!question) {
    setInfo("질문을 입력해주세요!", "error");
    els.question?.focus();
    return;
  }
  
  // UI 상태 변경
  setInfo("🤔 모델이 생각 중입니다... (GPU: 빠름, CPU: 1-2분 소요)", "info");
  
  if (els.answer) {
    els.answer.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>답변을 생성하는 중입니다...</p>
      </div>
    `;
  }
  
  if (els.contexts) {
    els.contexts.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>컨텍스트를 검색하는 중입니다...</p>
      </div>
    `;
  }
  
  if (els.contextCount) {
    els.contextCount.textContent = "검색 중...";
  }
  
  if (els.sendBtn) {
    els.sendBtn.disabled = true;
    const btnText = els.sendBtn.querySelector(".btn-text");
    if (btnText) {
      btnText.textContent = "생성 중...";
    }
  }
  
  try {
    const startTime = Date.now();
    
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k,
        max_new_tokens,
        filter_keyword,
        context_only,
        debug,
      }),
    });
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    
    if (!res.ok) {
      const errorText = await res.text();
      setInfo(`API 요청 실패 (${res.status}): ${errorText}`, "error");
      setStatusError("요청 실패");
      
      if (els.answer) {
        els.answer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">❌</div>
            <p>서버 오류가 발생했습니다.<br/>상태 코드: ${res.status}</p>
          </div>
        `;
      }
      
      if (els.contexts) {
        els.contexts.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <p>컨텍스트를 불러올 수 없습니다.</p>
          </div>
        `;
      }
      return;
    }
    
    const data = await res.json();
    setInfo(`✅ 응답 완료 (${elapsed}초)`, "success");
    
    // 답변 표시
    if (els.answer) {
      if (data.answer) {
        if (context_only && data.answer.includes("컨텍스트만")) {
          els.answer.innerHTML = `
            <div style="padding: 12px; background: var(--accent-soft); border-radius: var(--radius-md); margin-bottom: 12px;">
              <strong>ℹ️ 컨텍스트 전용 모드</strong><br/>
              답변은 생성되지 않았습니다. 아래 컨텍스트를 확인하세요.
            </div>
            <div style="white-space: pre-wrap;">${data.answer}</div>
          `;
        } else {
          els.answer.textContent = data.answer;
        }
      } else {
        els.answer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <p>답변이 비어 있습니다.<br/>백엔드 로그를 확인하세요.</p>
          </div>
        `;
      }
    }
    
    // 컨텍스트 표시
    renderContexts(data.contexts || [], data.used_keyword || null);
    
  } catch (err) {
    console.error("[sendQuestion 오류]", err);
    setInfo(`❌ 요청 중 오류: ${err.message || "알 수 없는 오류"}`, "error");
    setStatusError("에러 발생");
    
    if (els.answer) {
      els.answer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">❌</div>
          <p>오류가 발생했습니다:<br/>${err.message || "네트워크 오류"}</p>
        </div>
      `;
    }
    
    if (els.contexts) {
      els.contexts.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <p>컨텍스트를 불러올 수 없습니다.</p>
        </div>
      `;
    }
  } finally {
    if (els.sendBtn) {
      els.sendBtn.disabled = false;
      const btnText = els.sendBtn.querySelector(".btn-text");
      if (btnText) {
        btnText.textContent = "질문 보내기";
      }
    }
  }
}

function renderContexts(contexts, usedKeyword = null) {
  if (!els.contexts) return;
  
  els.contexts.innerHTML = "";
  
  if (els.contextCount) {
    els.contextCount.textContent = `${contexts.length} 개`;
  }
  
  if (!contexts.length) {
    els.contexts.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📄</div>
        <p>검색된 컨텍스트가 없습니다.<br/>질문을 더 구체적으로 바꿔보세요.</p>
      </div>
    `;
    return;
  }
  
  const list = document.createElement("div");
  list.className = "context-cards";
  
  // 사용된 키워드 정보 표시
  if (usedKeyword) {
    const filterInfo = document.createElement("div");
    filterInfo.className = "context-filter-info";
    filterInfo.innerHTML = `🔍 <strong>필터 적용:</strong> ${usedKeyword}`;
    list.appendChild(filterInfo);
  }
  
  // 컨텍스트 카드 생성
  contexts.forEach((c, idx) => {
    const card = document.createElement("article");
    card.className = "context-card";
    
    const score = typeof c.score === "number" ? c.score.toFixed(3) : "N/A";
    const kw = c.keyword || "";
    
    // 점수에 따라 색상 클래스 결정
    let scoreClass = "score-low";
    if (typeof c.score === "number") {
      if (c.score > 0.8) scoreClass = "score-high";
      else if (c.score > 0.6) scoreClass = "score-medium";
    }
    
    card.innerHTML = `
      <header>
        <span class="ctx-index">#${idx + 1}</span>
        <span class="ctx-score ${scoreClass}">score=${score}</span>
        ${kw ? `<span class="ctx-keyword">${kw}</span>` : ""}
      </header>
      <p class="ctx-text">${c.text}</p>
    `;
    
    list.appendChild(card);
  });
  
  els.contexts.appendChild(list);
}

// ============================================
// 이벤트 바인딩 (init 함수로 이동됨)
// ============================================

// ============================================
// 초기화
// ============================================
function init() {
  initElements();
  loadApiStatusAndKeywords();
  
  // 슬라이더 라벨 동기화
  if (els.topkInput && els.topkLabel) {
    els.topkInput.addEventListener("input", () => {
      els.topkLabel.textContent = els.topkInput.value;
    });
  }
  
  if (els.maxTokensInput && els.maxTokensLabel) {
    els.maxTokensInput.addEventListener("input", () => {
      els.maxTokensLabel.textContent = els.maxTokensInput.value;
    });
  }
  
  // 이벤트 바인딩
  if (els.sendBtn) {
    els.sendBtn.addEventListener("click", sendQuestion);
  }
  
  if (els.question) {
    els.question.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        sendQuestion();
      }
    });
  }
  
  if (els.clearBtn) {
    els.clearBtn.addEventListener("click", () => {
      if (els.answer) {
        els.answer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">💭</div>
            <p>왼쪽에 질문을 입력하고 전송 버튼을 눌러주세요.</p>
          </div>
        `;
      }
      
      if (els.contexts) {
        els.contexts.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">📄</div>
            <p>검색된 컨텍스트가 여기에 표시됩니다.</p>
          </div>
        `;
      }
      
      if (els.contextCount) {
        els.contextCount.textContent = "0 개";
      }
      
      setInfo("");
    });
  }
  
  if (els.refreshStatsBtn) {
    els.refreshStatsBtn.addEventListener("click", async () => {
      els.refreshStatsBtn.disabled = true;
      const originalText = els.refreshStatsBtn.textContent;
      els.refreshStatsBtn.textContent = "🔄";
      
      try {
        await loadKeywordsFallback();
      } finally {
        els.refreshStatsBtn.disabled = false;
        els.refreshStatsBtn.textContent = originalText;
      }
    });
  }
}

// 페이지 로드 시 초기화
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
