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
  $("predictionDescription").textContent = "The combined result and the neural-network prediction will appear here after analysis.";
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
    if (!modelData || !modelData.pca || !modelData.lda || !modelData.mlp) throw new Error("Model file is incomplete");
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
// Match the Python training pipeline: grayscale first, then Lanczos-3 resize,
// then scale byte intensities to [0, 1]. Canvas' default drawImage filter is
// browser-dependent and does not match PIL's Lanczos resampling.
function lanczosKernel(x) {
  x = Math.abs(x);
  if (x < 1e-8) return 1;
  if (x >= 3) return 0;
  const pix = Math.PI * x;
  return (Math.sin(pix) / pix) * (Math.sin(pix / 3) / (pix / 3));
}
function resizeLanczos1D(source, sourceLength, targetLength, sourceStride, sourceOffset) {
  const output = new Float32Array(targetLength);
  const scale = sourceLength / targetLength;
  const filterScale = Math.max(1, scale);
  const support = 3 * filterScale;
  for (let target = 0; target < targetLength; target++) {
    const center = (target + 0.5) * scale;
    const first = Math.max(0, Math.ceil(center - support - 0.5));
    const last = Math.min(sourceLength - 1, Math.floor(center + support - 0.5));
    let weightSum = 0;
    let value = 0;
    for (let sample = first; sample <= last; sample++) {
      const weight = lanczosKernel((sample + 0.5 - center) / filterScale);
      value += source[sourceOffset + sample * sourceStride] * weight;
      weightSum += weight;
    }
    output[target] = weightSum ? value / weightSum : 0;
  }
  return output;
}
function preprocessImage(image, size) {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = image.naturalWidth;
  sourceCanvas.height = image.naturalHeight;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceContext.drawImage(image, 0, 0);
  const rgba = sourceContext.getImageData(0, 0, image.naturalWidth, image.naturalHeight).data;
  const grayscale = new Float32Array(image.naturalWidth * image.naturalHeight);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    // PIL Image.convert("L") uses the same standard luma weighting.
    grayscale[p] = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
  }

  // Resize each row first, then each column (separable 2-D Lanczos).
  const sourceWidth = image.naturalWidth;
  const sourceHeight = image.naturalHeight;
  const horizontal = new Float32Array(sourceHeight * size);
  for (let y = 0; y < sourceHeight; y++) {
    const row = resizeLanczos1D(grayscale, sourceWidth, size, 1, y * sourceWidth);
    horizontal.set(row, y * size);
  }
  const resized = new Float32Array(size * size);
  for (let x = 0; x < size; x++) {
    const column = resizeLanczos1D(horizontal, sourceHeight, size, size, x);
    for (let y = 0; y < size; y++) resized[y * size + x] = column[y];
  }
  const vector = new Array(size * size);
  for (let i = 0; i < vector.length; i++) {
    vector[i] = Math.max(0, Math.min(255, Math.round(resized[i]))) / 255;
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
  const scale = Math.min(1, 1000 / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  const source = ctx.getImageData(0, 0, width, height);
  const rgba = source.data, total = width * height, stride = width + 1;
  const gray = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    const p = i * 4;
    gray[i] = Math.round(0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]);
  }

  // Local mean lets the detector adapt to lighting; use a stricter dark-pixel
  // threshold so ordinary concrete texture does not connect the whole image.
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 1; y <= height; y++) {
    let row = 0;
    for (let x = 1; x <= width; x++) {
      row += gray[(y - 1) * width + x - 1];
      integral[y * stride + x] = integral[(y - 1) * stride + x] + row;
    }
  }
  const radius = Math.max(7, Math.round(Math.min(width, height) * 0.025));
  const mask = new Uint8Array(total);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const x1 = Math.max(0, x - radius), x2 = Math.min(width, x + radius + 1);
    const y1 = Math.max(0, y - radius), y2 = Math.min(height, y + radius + 1);
    const area = (x2 - x1) * (y2 - y1);
    const sum = integral[y2 * stride + x2] - integral[y1 * stride + x2] - integral[y2 * stride + x1] + integral[y1 * stride + x1];
    const contrast = sum / area - gray[y * width + x];
    // Ignore a narrow frame where image borders can create false crack lines,
    // and the common bottom-left camera/lens overlay area.
    const edge = Math.min(width, height) * 0.012;
    const inCameraOverlay = x < width * 0.075 && y > height * 0.86;
    if (x > edge && x < width - edge && y > edge && y < height - edge &&
        !inCameraOverlay && gray[y * width + x] < 112 && contrast > 25) {
      mask[y * width + x] = 1;
    }
  }

  // Connected components on the un-dilated mask preserve separate crack paths.
  const visited = new Uint8Array(total), regions = [];
  const minArea = Math.max(7, Math.round(total * 0.000008));
  const minLength = Math.max(9, Math.round(Math.min(width, height) * 0.018));
  for (let start = 0; start < total; start++) {
    if (!mask[start] || visited[start]) continue;
    const stack = [start], points = [];
    visited[start] = 1;
    let minX = width, minY = height, maxX = 0, maxY = 0, sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
    while (stack.length) {
      const at = stack.pop(), x = at % width, y = Math.floor(at / width);
      points.push(at);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      sumX += x; sumY += y; sumXX += x*x; sumYY += y*y; sumXY += x*y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const ni = ny * width + nx;
        if (mask[ni] && !visited[ni]) { visited[ni] = 1; stack.push(ni); }
      }
    }
    const boxW = maxX - minX + 1, boxH = maxY - minY + 1;
    const length = Math.max(boxW, boxH), thickness = Math.max(1, Math.min(boxW, boxH));
    const elongation = length / thickness;
    const n = points.length;
    const vx = sumXX / n - (sumX / n) ** 2;
    const vy = sumYY / n - (sumY / n) ** 2;
    const cov = sumXY / n - (sumX / n) * (sumY / n);
    const angle = 0.5 * Math.atan2(2 * cov, vx - vy);
    // Keep prominent thick cracks as well as thin elongated hairline segments.
    if (n >= minArea && length >= minLength && (elongation >= 1.45 || n >= total * 0.00012)) {
      regions.push({ points, minX, minY, maxX, maxY, length, angle, area: n });
    }
  }

  // Join only close, similarly oriented fragments; never use a large radius
  // that can combine unrelated texture into one image-sized box.
  const merged = [];
  for (const region of regions.sort((a,b) => b.area-a.area)) {
    const match = merged.find(m => {
      const gapX = Math.max(0, region.minX - m.maxX, m.minX - region.maxX);
      const gapY = Math.max(0, region.minY - m.maxY, m.minY - region.maxY);
      const angleDiff = Math.abs(Math.atan2(Math.sin(region.angle-m.angle), Math.cos(region.angle-m.angle)));
      return Math.hypot(gapX,gapY) < Math.max(5, Math.min(width,height)*0.012) && angleDiff < 0.45;
    });
    if (match) {
      match.points.push(...region.points);
      match.minX=Math.min(match.minX,region.minX); match.minY=Math.min(match.minY,region.minY);
      match.maxX=Math.max(match.maxX,region.maxX); match.maxY=Math.max(match.maxY,region.maxY);
      match.area+=region.area;
    } else merged.push({...region,points:[...region.points]});
  }

  // Thin the binary crack mask with Zhang-Suen skeletonization. This keeps
  // the red trace close to the crack centerline instead of painting it solid.
  const skeleton = new Uint8Array(total);
  for (let i = 0; i < total; i++) if (mask[i]) skeleton[i] = 1;
  let changed = true, passes = 0;
  while (changed && passes < 24) {
    changed = false; passes++;
    for (let phase = 0; phase < 2; phase++) {
      const remove = [];
      for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
        const at = y * width + x;
        if (!skeleton[at]) continue;
        const p2=skeleton[at-width], p3=skeleton[at-width+1], p4=skeleton[at+1], p5=skeleton[at+width+1];
        const p6=skeleton[at+width], p7=skeleton[at+width-1], p8=skeleton[at-1], p9=skeleton[at-width-1];
        const neighbors=[p2,p3,p4,p5,p6,p7,p8,p9];
        const count=neighbors.reduce((a,b)=>a+b,0);
        let transitions=0;
        for(let k=0;k<8;k++) if(!neighbors[k] && neighbors[(k+1)%8]) transitions++;
        if(count<2 || count>6 || transitions!==1) continue;
        const conditionA = phase===0 ? (p2*p4*p6===0 && p4*p6*p8===0) : (p2*p4*p8===0 && p2*p6*p8===0);
        if(conditionA) remove.push(at);
      }
      if(remove.length){ changed=true; for(const at of remove) skeleton[at]=0; }
    }
  }

  // Keep only skeleton pixels belonging to accepted components; blend a
  // one-pixel red trace so the concrete and original crack remain visible.
  const output = ctx.getImageData(0,0,width,height);
  const out = output.data;
  const accepted = new Uint8Array(total);
  for (const region of merged) for (const at of region.points) accepted[at]=1;
  for(let i=0;i<total;i++) if(skeleton[i] && accepted[i]){
    const p=i*4, alpha=0.78;
    out[p]=Math.round(out[p]*(1-alpha)+255*alpha);
    out[p+1]=Math.round(out[p+1]*(1-alpha)+35*alpha);
    out[p+2]=Math.round(out[p+2]*(1-alpha)+45*alpha);
  }
  ctx.putImageData(output,0,0);

  const coverage = merged.reduce((s,r)=>s+r.area,0)/total;
  const largestExtent = merged.reduce((m,r)=>Math.max(m,r.length/Math.max(width,height)),0);
  $("markingStatus").textContent = merged.length ? "CRACK-LIKE PATHS HIGHLIGHTED" : "NO REGIONS FOUND";
  $("crackCount").textContent = String(merged.length);
  $("crackCountText").textContent = merged.length
    ? "Thin red centerline traces highlight " + merged.length + " candidate crack-like paths. Large connected cracks and smaller branches are retained; isolated texture and border artifacts are filtered where possible. This is not a verified physical crack count."
    : "No clear crack-like paths were identified. This does not guarantee the surface is crack-free.";
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
    const modelSaysCracked = bestIndex === modelData.cracked_class_index;
    const neuralNetworkScore = probabilities[bestIndex] * 100;
    const score = ldaDecision(features, modelData.lda);

    // Combine two independent signals for the headline result:
    // the trained neural network prediction and the visual crack-like-region detector.
    // Either positive signal flags the image; LDA remains a separate diagnostic below.
    const imageFindings = markSuspectedCracks(preview);
    const visualDetectorSaysCracked = imageFindings.count > 0;
    const isCracked = modelSaysCracked || visualDetectorSaysCracked;

    $("predictionLabel").textContent = isCracked ? "Cracks Detected" : "No Cracks Detected";
    $("predictionStatus").textContent = "COMPLETE";
    $("predictionIcon").textContent = isCracked ? "⌁" : "✓";
    $("predictionDescription").textContent = isCracked
      ? "Combined system result: Positive — crack presence flagged by " + [modelSaysCracked ? "the neural network" : "", visualDetectorSaysCracked ? "the visual detector" : ""].filter(Boolean).join(" and ") + ". Actual neural-network prediction: " + modelData.classes[bestIndex] + " (" + neuralNetworkScore.toFixed(1) + "% score for that predicted class). LDA is shown separately as a diagnostic."
      : "Combined system result: Negative — neither the neural network nor the visual detector identified a crack pattern. Actual neural-network prediction: " + modelData.classes[bestIndex] + " (" + neuralNetworkScore.toFixed(1) + "% score for that predicted class). This image-based result is not proof that the surface is crack-free.";
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
      $("guidanceText").textContent = "The neural network and/or visual detector flagged a crack-like pattern. LDA may disagree and is displayed separately. This image result cannot determine whether plastering is enough or whether structural repair is needed. An on-site inspection is needed to check crack width, depth, location, movement, and the wall's role in the building.";
    } else {
      $("guidanceTitle").textContent = "No crack pattern detected by the model";
      $("guidanceText").textContent = "The model classified this image as non-cracked. This does not certify that the wall is safe: hidden defects, small cracks, poor image quality, or surfaces unlike the training images may be missed. Inspect the wall if there are visible warning signs or concerns.";
    }
    // This percentage is the neural network's own class score, not a calibrated
    // confidence for the combined crack-detection decision.
    $("confidenceValue").textContent = neuralNetworkScore.toFixed(1) + "%";
    $("confidenceBar").style.width = neuralNetworkScore.toFixed(1) + "%";
    // sklearn's binary LDA decision_function is positive for classes[1]
    // and negative for classes[0]. Keep the raw score visible, but label it.
    const ldaClasses = modelData.lda.classes || modelData.classes;
    const ldaClassIndex = score >= 0 ? 1 : 0;
    const ldaClassName = ldaClasses[ldaClassIndex] || "Unknown class";
    $("ldaScore").textContent = Number(score).toFixed(3) + " · " + ldaClassName;
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