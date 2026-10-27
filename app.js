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
    $("guidanceStatus").textContent = "PRELIMINARY GUIDANCE";
    // Prototype-only heuristic: this is deliberately separate from neural-network confidence.
    // It is not trained or validated as a wall-quality metric.
    const healthScore = isCracked ? 55 : 85;
    $("healthScore").innerHTML = healthScore + "<small>/100</small>";
    $("healthGrade").textContent = isCracked ? "Needs inspection" : "No crack pattern detected";
    $("healthMeterFill").style.width = healthScore + "%";
    $("healthMeterFill").classList.toggle("health-caution", isCracked);
    $("healthSummary").textContent = isCracked
      ? "A crack pattern was detected, so this prototype assigns a lower surface indicator. Crack severity and structural condition are not measured."
      : "No crack pattern was detected, so this prototype shows a higher surface indicator. This is not proof of good wall health or safety.";
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