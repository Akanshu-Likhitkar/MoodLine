(() => {
  "use strict";

  const EMOJI = {
    sadness: "😢",
    joy: "😄",
    love: "❤️",
    anger: "😠",
    fear: "😨",
    surprise: "😲",
  };

  const el = {
    statusDot: document.getElementById("statusDot"),
    serverStatusText: document.getElementById("serverStatusText"),
    textInput: document.getElementById("textInput"),
    charCount: document.getElementById("charCount"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    errorMsg: document.getElementById("errorMsg"),
    resultSection: document.getElementById("resultSection"),
    emotionWord: document.getElementById("emotionWord"),
    emotionEmoji: document.getElementById("emotionEmoji"),
    confidenceText: document.getElementById("confidenceText"),
    echoedText: document.getElementById("echoedText"),
    barsContainer: document.getElementById("barsContainer"),
  };

  let modelReady = false;

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();

      modelReady = !!data.model_loaded;
      if (modelReady) {
        setStatus("live", "model ready");
      } else {
        setStatus("warming", "waking the model up…");
        setTimeout(checkHealth, 3000);
      }
    } catch (e) {
      setStatus("down", "can't reach the server");
      setTimeout(checkHealth, 5000);
    }
    syncButtonState();
  }

  function setStatus(kind, text) {
    el.statusDot.style.backgroundColor = kind === "live" ? "#54b25d" : (kind === "warming" ? "#e2c044" : "#ff5555");
    el.serverStatusText.textContent = text;
  }

  el.textInput.addEventListener("input", () => {
    el.charCount.textContent = el.textInput.value.length;
    syncButtonState();
  });

  function syncButtonState() {
    const hasText = el.textInput.value.trim().length > 0;
    el.analyzeBtn.disabled = !hasText || !modelReady;
  }

  el.analyzeBtn.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    const text = el.textInput.value.trim();
    if (!text || !modelReady) return;

    hideError();
    el.analyzeBtn.disabled = true;
    el.analyzeBtn.querySelector(".btn-label").textContent = "Reading…";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error("Request failed.");

      const data = await res.json();
      renderResult(data, text);
    } catch (err) {
      el.analyzeBtn.querySelector(".btn-label").textContent = "Read the mood";
      syncButtonState();
      showError(err.message || "Something went wrong. Try again.");
    }
  }

  function renderResult(data, originalText) {
    const emotion = data.predicted_emotion;
    const emoji = EMOJI[emotion] || "🙂";

    el.analyzeBtn.querySelector(".btn-label").textContent = "Read the mood";
    syncButtonState();

    el.emotionWord.textContent = capitalize(emotion);
    el.emotionEmoji.textContent = emoji;
    
    // Convert confidence to float and handle NaN protection
    const confValue = parseFloat(data.confidence) || 0;
    el.confidenceText.textContent = `${(confValue * 100).toFixed(1)}% confidence`;
    
    el.echoedText.textContent = `“${originalText}”`;

    renderBars(data.all_probabilities);
    el.resultSection.hidden = false;
  }

  function renderBars(probs) {
    // Sort probabilities highest to lowest
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    el.barsContainer.innerHTML = "";

    entries.forEach(([label, value]) => {
      // Ensure value is treated as number
      const floatVal = parseFloat(value) || 0;
      const pct = floatVal * 100;
      
      const row = document.createElement("div");
      row.className = "bar-row";
      
      row.innerHTML = `
        <span class="bar-label">
            <span>${EMOJI[label] || ""}</span> 
            ${label}
        </span>
        <div class="bar-track">
            <div class="bar-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="bar-pct">${pct.toFixed(1)}%</span>
      `;
      el.barsContainer.appendChild(row);
    });
  }

  function showError(msg) {
    el.errorMsg.textContent = msg;
    el.errorMsg.hidden = false;
  }
  function hideError() { el.errorMsg.hidden = true; }
  function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  checkHealth();
})();