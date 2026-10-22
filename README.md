# BuildSense AI

**BuildSense AI** is an educational computer-vision project for classifying concrete surface images into two labelled categories (for example, cracked and non-cracked). It demonstrates three FOML techniques in a single workflow:

- **PCA (Principal Component Analysis):** reduces grayscale pixel dimensions and reports retained variance.
- **LDA (Linear Discriminant Analysis):** learns a supervised projection/classifier from class labels.
- **Neural Network (MLPClassifier):** predicts a class from PCA-reduced features.

The web interface is a static HTML/CSS/JavaScript site. Training happens locally in Python; the exported numerical parameters are loaded in the browser for inference. This architecture avoids needing a Python inference server on Vercel.

> **Important limitation:** this is an experimental image classifier, not a structural inspection, engineering diagnosis, or building-safety tool. Performance depends on the training data and the uploaded image conditions.

## Repository layout

```text
BuildSense-AI/
├── index.html
├── styles.css
├── app.js
├── training/train.py
├── training/requirements.txt
├── vercel.json
├── data/concrete_cracks/raw/   # local dataset; do not commit
└── public/model.json           # generated after training; required for predictions
```

## 1. Get the labelled image dataset

Use the genuine research dataset **Concrete Crack Images for Classification**:

https://data.mendeley.com/datasets/5y9wdsg2zt/2

Review the dataset page and licence before use. Download and extract the archive. The training script expects a directory with exactly two class subfolders containing images, for example:

```text
data/concrete_cracks/raw/
├── Negative/
│   ├── image1.jpg
│   └── ...
└── Positive/
    ├── image1.jpg
    └── ...
```

If your extracted folder names or structure differ, do not guess. Point `--data-dir` to the folder directly containing the two class folders. The script checks the structure and reports an error if it cannot identify exactly two image classes.

## 2. Set up Python locally

Python 3.11 or 3.12 is recommended for compatibility with scientific Python packages.

Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r training/requirements.txt
```

If you already have a working virtual environment with compatible dependencies, you can use it instead.

## 3. Train and evaluate

Start with a small run first:

```powershell
python training/train.py --data-dir data/concrete_cracks/raw --max-per-class 250 --components 30
```

Inspect the output: loaded class counts, PCA retained variance, LDA metrics, neural-network metrics, classification report and confusion matrix. If that run succeeds, try a larger experiment:

```powershell
python training/train.py --data-dir data/concrete_cracks/raw --max-per-class 1200 --components 60
```

The script uses a stratified train/test split and a fixed random seed. It writes `public/model.json`, which contains PCA, LDA and MLP parameters plus evaluation metrics. The web page uses the PCA and MLP parameters for the displayed prediction; the LDA score is shown separately as an educational diagnostic.

**Model export details:** the browser preprocesses images as grayscale, resized to 64×64 and scaled to [0, 1]. This must match training preprocessing. The generated JSON can be a few megabytes. Commit it only after you have trained and inspected the result.

### Dataset and evaluation caveats

- Do not include the dataset images themselves in GitHub.
- The script samples at most the requested number of images per class so the first experiment can run on a typical laptop.
- A random split may overestimate generalization if related crops or near-duplicates occur across splits. For a stronger research result, use a source-aware split if the dataset supplies source identities, and report that limitation in the project report.
- Accuracy is not proof of real-world reliability. Inspect class-wise precision/recall, confusion matrix and misclassified examples.

## 4. Run the website locally

The web app is static. After training generates `public/model.json`, run a static server from the repository root. For example:

```powershell
python -m http.server 8000
```

Open http://localhost:8000. The model JSON is served from `/public/model.json` only if the file is at the web root; to keep this minimal static layout, **place or copy the generated file to the repository root as `model.json` and update the fetch path in `app.js` to `/model.json`**, or use the Vercel static output configuration described below.

### Recommended simple static layout for Vercel

For the simplest deployment, put the generated file at the repository root as `model.json` (same level as `index.html`). The app already fetches `/model.json`. To do this after training:

```powershell
Copy-Item public/model.json model.json
```

You may keep the generated file in `public/` as your training output, but copy it to the root for this static-site setup.

## 5. Deploy to Vercel

1. Push/commit the repository to GitHub.
2. In Vercel, choose **Add New → Project** and import `BuildSense-AI`.
3. Framework preset: **Other**.
4. Build command: leave empty.
5. Output directory: leave empty / project root.
6. Deploy.
7. After training locally, copy `public/model.json` to the repository root as `model.json`, commit and push it. Vercel redeploys automatically.

If you deploy before adding a trained `model.json`, the interface should still load, but prediction is intentionally disabled and the app explains that the model has not been trained yet.

## 6. Current scope

- Upload or drag-and-drop a concrete image.
- Local browser image preview and grayscale preprocessing.
- Neural-network prediction when a trained model file is available.
- PCA component and retained-variance display.
- LDA decision score display.
- Training metrics printed to the terminal.

Potential next improvements after the initial run works: a PCA reconstruction panel, a proper two-dimensional training-set scatter plot, threshold calibration, class-wise metric charts and tests for browser/Python preprocessing parity.

## Data and attribution

Dataset: *Concrete Crack Images for Classification*, hosted on Mendeley Data. Cite the dataset authors and version according to the official record in any academic report and follow the licence terms on the dataset page.

## Safety statement

BuildSense AI classifies visible image patterns for educational demonstration only. It cannot determine structural integrity or replace inspection by a qualified civil/structural engineer.
