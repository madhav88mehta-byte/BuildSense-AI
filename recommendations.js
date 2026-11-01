(() => {
  const predictionLabel = document.getElementById("predictionLabel");
  const healthScore = document.getElementById("healthScore");
  const recommendationGrid = document.querySelector(".guidance-grid");

  if (!predictionLabel || !healthScore || !recommendationGrid) return;

  function updateRepairRecommendations() {
    const noCrackDetected = predictionLabel.textContent.includes("NO CRACK DETECTED");
    const scoreMatch = healthScore.textContent.match(/\d+/);
    const score = scoreMatch ? Number(scoreMatch[0]) : NaN;
    const hideRecommendations = noCrackDetected && Number.isFinite(score) && score > 85;

    recommendationGrid.style.display = hideRecommendations ? "none" : "";
    recommendationGrid.setAttribute("aria-hidden", String(hideRecommendations));
  }

  const observer = new MutationObserver(updateRepairRecommendations);
  observer.observe(predictionLabel, { childList: true, characterData: true, subtree: true });
  observer.observe(healthScore, { childList: true, characterData: true, subtree: true });
  updateRepairRecommendations();
})();
