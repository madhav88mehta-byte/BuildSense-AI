const $ = (id) => document.getElementById(id);
const input = $("imageInput");
const dropzone = $("dropzone");
const preview = $("imagePreview");
const analyzeButton = $("analyzeButton");
let selectedFile = null;
let modelData = null;
let previewObjectUrl = null;

function resetPhoto() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  selectedFile = null;
  input.value = "";
  preview.removeAttribute("src");
  preview.hidden = true;
  $( "previewStage" ).querySelector(".empty-preview").classList.remove("hidden");
  $("fileName").textContent = "No image selected";
  $("imageDimensions").textContent = "— × —";
  $("imageState").textContent = "WAITING FOR INPUT";
  $("resetButton").disabled = true;
  analyzeButton.disabled = true;

  $("predictionLabel").textContent = "Awaiting image";
  $("predictionStatus").textContent = "NO RESULT";
  $("predictionIcon").textContent = "⌁";
  $("predictionDescription").textContent = "Choose a concrete image and run the trained model to see its output.";
  $("confidenceValue").textContent = "—";
  $("confidenceBar").style.width = "0%";
  $("ldaScore").textContent = "—";

  $("pcaStatus").textContent = "WAITING";
  $("pcaComponents").textContent = "—";
  $("pcaVariance").textContent = "—";
  $("pcaPlot").querySelectorAll(".pca-point,.plot-placeholder").forEach((el) => el.remove());
  const placeholder = document.createElement("div");
  placeholder.className = "plot-placeholder";
  placeholder.textContent = "PCA projection appears after analysis";
  $("pcaPlot").appendChild(placeholder);

  $("guidanceStatus").textContent = "AWAITING ANALYSIS";
  $("healthScore").innerHTML = "—<small>/100</small>";
  $("healthGrade").textContent = "Awaiting image";
  $("healthMeterFill").style.width = "0%";
  $("healthMeterFill").classList.remove("health-caution");
  $("healthSummary").textContent = "Run an image analysis to get a preliminary surface-condition indication.";
  $("guidanceTitle").textContent = "A practical next step will appear here";
  $("guidanceText").textContent = "Upload a surface image and select Analyze surface. This prototype classifies crack presence and shows a provisional surface-condition indicator; it does not measure crack severity or determine structural safety.";
  clearCrackMarkings();
}

$("resetButton").addEventListener("click", resetPhoto);

async function loadModel() {
  try {
    const response = await fetch("/model.json", { cache: "no-store" });
    if (!response.ok) throw new Error("No trained model file");
    modelData = await response.json();
    if (!modelData || !modelData.pca || !modelData.mlp) throw new Error("Model file is incomplete");
    $("modelStatus").textContent = "Trained model loaded · image analysis is ready.";
    $("modelNotice").classList.add("hidden");
    analyzeButton.disabled = !selectedFile;
  } catch (_) {
    $("modelStatus").textContent = "Model not trained yet. Follow README.md to train and export it.";
    $("modelNotice").classList.remove("hidden");
    analyzeButton.disabled = true;
  }
}

function setFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith("image/")) {
    alert("Please select an image file (JPG, PNG or WEBP).");
    return;
  }
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  selectedFile = file;
  previewObjectUrl = URL.createObjectURL(file);
  preview.onload = () => {
    $("imageDimensions").textContent = preview.naturalWidth + " × " + preview.naturalHeight;
  };
  preview.src = previewObjectUrl;
  preview.hidden = false;
  dropzone.classList.remove("dragging");
  $("previewStage").querySelector(".empty-preview").classList.add("hidden");
  $("fileName").textContent = file.name;
  $("imageState").textContent = "IMAGE LOADED";
  $("resetButton").disabled = false;
  analyzeButton.disabled = !modelData;
  $("resultsSection").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
input.addEventListener("change", (event) => setFile(event.target.files[0]));
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
function denseForward(inputVector, weights, biases, activation) {
  const out = new Array(biases.length);
  for (let j = 0; j < biases.length; j++) {
    let total = biases[j];
    const column = weights.map((row) => row[j]);
    for (let i = 0; i < inputVector.length; i++) total += inputVector[i] * column[i];
    out[j] = activation === "relu" ? Math.max(0, total) : total;
  }
  return out;
}
function predictMLP(features, mlp) {
  let current = features;
  for (let layer = 0; layer < mlp.coefs.length; layer++) {
    const weights = mlp.coefs[layer];
    const biases = mlp.intercepts[layer];
    const next = new Array(biases.length);
    for (let j = 0; j < biases.length; j++) {
      let total = biases[j];
      for (let i = 0; i < current.length; i++) total += current[i] * weights[i][j];
      next[j] = layer < mlp.coefs.length - 1 ? Math.max(0, total) : total;
    }
    current = next;
  }
  // scikit-learn's MLPClassifier uses one sigmoid output for binary labels.
  // Convert that probability into [class 0, class 1]. Multiclass exports use softmax.
  if (current.length === 1) {
    const z = Math.max(-40, Math.min(40, current[0]));
    const p1 = 1 / (1 + Math.exp(-z));
    return [1 - p1, p1];
  }
  return softmax(current);
}
function preprocessImage(image, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const rgba = context.getImageData(0, 0, size, size).data;
  const vector = new Array(size * size);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    const gray = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    vector[p] = gray / 255;
  }
  return vector;
}
function projectPCA(vector, pca) {
  const centered = vector.map((v, i) => v - pca.mean[i]);
  const result = new Array(pca.components.length);
  for (let c = 0; c < pca.components.length; c++) {
    const component = pca.components[c];
    let sum = 0;
    for (let i = 0; i < centered.length; i++) sum += centered[i] * component[i];
    result[c] = sum;
  }
  return result;
}
function ldaDecision(features, lda) {
  let score = lda.intercept[0];
  for (let i = 0; i < features.length; i++) score += features[i] * lda.coef[0][i];
  return score;
}
function markSuspectedCracks(image) {
  const canvas = $("markingCanvas");
  const maxSide = 900;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const data = pixels.data;
  const total = width * height;
  const gray = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    const p = i * 4;
    gray[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }

  // Integral image gives a stable local brightness estimate at multiple image sizes.
  const stride = width + 1;
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      rowSum += gray[(y - 1) * width + (x - 1)];
      integral[y * stride + x] = integral[(y - 1) * stride + x] + rowSum;
    }
  }
  const radius = Math.max(5, Math.round(Math.min(width, height) * 0.018));
  const mask = new Uint8Array(total);
  for (let y = 0; y < height; y++) {
    const y1 = Math.max(0, y - radius), y2 = Math.min(height, y + radius + 1);
    for (let x = 0; x < width; x++) {
      const x1 = Math.max(0, x - radius), x2 = Math.min(width, x + radius + 1);
      const area = (x2 - x1) * (y2 - y1);
      const sum = integral[y2 * stride + x2] - integral[y1 * stride + x2] - integral[y2 * stride + x1] + integral[y1 * stride + x1];
      const localMean = sum / area;
      const darkness = localMean - gray[y * width + x];
      // Ignore bright texture; retain dark, locally contrasting crack pixels.
      if (gray[y * width + x] < 150 && darkness > 20) mask[y * width + x] = 1;
    }
  }

  // A single 3x3 closing pass joins tiny gaps along a continuous crack without
  // aggressively merging distant specks into giant circular regions.
  const dilated = new Uint8Array(total);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    let on = 0;
    for (let dy = -1; dy <= 1 && !on; dy++) for (let dx = -1; dx <= 1; dx++) if (mask[(y + dy) * width + x + dx]) { on = 1; break; }
    if (on) dilated[y * width + x] = 1;
  }
  const closed = new Uint8Array(total);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    let all = 1;
    for (let dy = -1; dy <= 1 && all; dy++) for (let dx = -1; dx <= 1; dx++) if (!dilated[(y + dy) * width + x + dx]) { all = 0; break; }
    if (all) closed[y * width + x] = 1;
  }

  const visited = new Uint8Array(total);
  const components = [];
  const minPixels = Math.max(10, Math.round(total * 0.000025));
  for (let start = 0; start < total; start++) {
    if (!closed[start] || visited[start]) continue;
    const stack = [start];
    visited[start] = 1;
    let count = 0, minX = width, minY = height, maxX = 0, maxY = 0;
    while (stack.length) {
      const at = stack.pop();
      const x = at % width, y = Math.floor(at / width);
      count++;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const ni = ny * width + nx;
        if (closed[ni] && !visited[ni]) { visited[ni] = 1; stack.push(ni); }
      }
    }
    const boxWidth = maxX - minX + 1, boxHeight = maxY - minY + 1;
    const longSide = Math.max(boxWidth, boxHeight);
    const shortSide = Math.max(1, Math.min(boxWidth, boxHeight));
    const elongation = longSide / shortSide;
    // Keep both small, line-like cracks and large connected crack branches;
    // reject isolated texture dots and broad shadow patches.
    if (count >= minPixels && longSide >= Math.max(12, Math.round(Math.min(width, height) * 0.025)) &&
        (elongation >= 1.7 || count >= total * 0.00035)) {
      components.push({ minX, minY, maxX, maxY, count, boxWidth, boxHeight, longSide });
    }
  }

  // Merge only boxes that overlap or nearly touch; do not merge by large radius.
  const merged = [];
  for (const item of components.sort((a, b) => b.count - a.count)) {
    let target = merged.find((m) =>
      item.minX <= m.maxX + 5 && item.maxX >= m.minX - 5 &&
      item.minY <= m.maxY + 5 && item.maxY >= m.minY - 5
    );
    if (target) {
      target.minX = Math.min(target.minX, item.minX); target.minY = Math.min(target.minY, item.minY);
      target.maxX = Math.max(target.maxX, item.maxX); target.maxY = Math.max(target.maxY, item.maxY);
      target.count += item.count;
    } else merged.push({ ...item });
  }

  // Tight rectangular outlines follow each candidate's actual footprint.
  ctx.lineWidth = Math.max(2, width / 420);
  ctx.strokeStyle = "#ff3d3d";
  ctx.fillStyle = "rgba(255, 45, 45, 0.07)";
  merged.forEach((region, i) => {
    const pad = Math.max(4, Math.round(Math.min(width, height) * 0.008));
    const x = Math.max(0, region.minX - pad), y = Math.max(0, region.minY - pad);
    const w = Math.min(width - x, region.maxX - region.minX + 1 + pad * 2);
    const h = Math.min(height - y, region.maxY - region.minY + 1 + pad * 2);
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#ff3d3d";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText("R" + (i + 1), x + 3, Math.max(13, y - 4));
    ctx.fillStyle = "rgba(255, 45, 45, 0.07)";
  });

  const crackPixels = merged.reduce((sum, r) => sum + r.count, 0);
  const largestExtent = merged.reduce((max, r) => Math.max(max, r.longSide / Math.max(width, height)), 0);
  const coverage = crackPixels / total;
  $("markingStatus").textContent = merged.length ? "CRACK-LIKE REGIONS MARKED" : "NO REGIONS FOUND";
  $("crackCount").textContent = String(merged.length);
  $("crackCountText").textContent = merged.length
    ? "Red outlines identify " + merged.length + " candidate regions. Larger connected regions are retained alongside smaller line-like candidates; these are not verified physical crack counts."
    : "No clear crack-like regions were identified in this image-processing pass. This does not guarantee the surface is crack-free.";
  $("markingEmpty").classList.add("hidden");
  return { count: merged.length, coverage, largestExtent, components: merged };
}
function clearCrackMarkings() {
  const canvas = $("markingCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = 0;
  canvas.height = 0;
  $("markingStatus").textContent = "AWAITING ANALYSIS";
  $("crackCount").textContent = "—";
  $("crackCountText").textContent = "The number of candidate regions will appear here after image processing.";
  $("markingEmpty").classList.remove("hidden");
}

function drawProjection(features, pca) {
  const plot = $("pcaPlot");
  plot.querySelectorAll(".pca-point,.plot-placeholder").forEach((el) => el.remove());
  const x = features[0] || 0;
  const y = features[1] || 0;
  const rangeX = pca.plot_range?.[0] || 8;
  const rangeY = pca.plot_range?.[1] || 8;
  const left = Math.max(5, Math.min(95, 50 + (x / rangeX) * 43));
  const top = Math.max(5, Math.min(95, 50 - (y / rangeY) * 43));
  const point = document.createElement("span");
  point.className = "pca-point";
  point.style.left = left + "%";
  point.style.top = top + "%";
  point.title = "Uploaded image PCA projection";
  plot.appendChild(point);
  $("pcaStatus").textContent = "PROJECTED";
  $("pcaComponents").textContent = String(pca.components.length);
  $("pcaVariance").textContent = (pca.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(1) + "%";
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || !modelData) return;
  analyzeButton.disabled = true;
  analyzeButton.querySelector("span:first-child").textContent = "Analyzing…";
  $("imageState").textContent = "PROCESSING";
  try {
    const features = projectPCA(preprocessImage(preview, modelData.image_size), modelData.pca);
    const probabilities = predictMLP(features, modelData.mlp);
    const bestIndex = probabilities.indexOf(Math.max(...probabilities));
    const label = modelData.classes[bestIndex];
    const confidence = probabilities[bestIndex] * 100;
    const score = ldaDecision(features, modelData.lda);
    $("predictionLabel").textContent = label;
    $("predictionStatus").textContent = "COMPLETE";
    $("predictionIcon").textContent = bestIndex === modelData.cracked_class_index ? "⌁" : "✓";
    $("predictionDescription").textContent = "Predicted from the uploaded image using PCA-reduced pixel features and the trained neural network.";
    const isCracked = bestIndex === modelData.cracked_class_index;
    const imageFindings = markSuspectedCracks(preview);
    $("guidanceStatus").textContent = "PRELIMINARY GUIDANCE";
    // Explainable image-based concern index, not a trained or engineering-grade score.
    // Penalize long connected regions and candidate coverage so major cracks matter more than speck count.
    const visualPenalty = Math.min(78,
      imageFindings.largestExtent * 58 +
      imageFindings.coverage * 900 +
      Math.min(imageFindings.count, 12) * 1.8 +
      (isCracked ? 12 : 0)
    );
    const healthScore = Math.max(12, Math.min(96, Math.round(96 - visualPenalty)));
    const concern = healthScore <= 45 ? "High visual concern" : healthScore <= 70 ? "Moderate visual concern" : healthScore <= 85 ? "Some visible concern" : "Few visual indicators";
    $("healthScore").innerHTML = healthScore + "<small>/100</small>";
    $("healthGrade").textContent = concern;
    $("healthMeterFill").style.width = healthScore + "%";
    $("healthMeterFill").classList.toggle("health-caution", healthScore <= 70);
    $("healthSummary").textContent = imageFindings.count
      ? "The visual indicator is reduced by " + imageFindings.count + " suspected crack-like regions, their image coverage, and the largest connected region. It is not a measurement of structural safety."
      : "No prominent crack-like regions were found by this preliminary image-processing pass. The indicator is not proof of good wall health or safety.";
    if (isCracked) {
      $("guidanceTitle").textContent = "Visible crack pattern suspected — inspection recommended";
      $("guidanceText").textContent = "The model classified this image as cracked. This result alone cannot tell whether plastering is enough or whether the wall needs structural repair or rebuilding. An on-site inspection is needed to check crack width, depth, location, movement, and the wall's role in the building.";
    } else {
      $("guidanceTitle").textContent = "No crack pattern detected by the model";
      $("guidanceText").textContent = "The model classified this image as non-cracked. This does not certify that the wall is safe: hidden defects, small cracks, poor image quality, or surfaces unlike the training images may be missed. Inspect the wall if there are visible warning signs or concerns.";
    }
    $("confidenceValue").textContent = confidence.toFixed(1) + "%";
    $("confidenceBar").style.width = confidence.toFixed(1) + "%";
    $("ldaScore").textContent = Number(score).toFixed(3);
    $("imageState").textContent = "ANALYSIS COMPLETE";
    drawProjection(features, modelData.pca);
    $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    alert("Analysis failed. Check that the trained model matches the app version. See the browser console for details.");
    $("imageState").textContent = "ANALYSIS ERROR";
  } finally {
    analyzeButton.disabled = !modelData || !selectedFile;
    analyzeButton.querySelector("span:first-child").textContent = "Analyze surface";
  }
});
loadModel();