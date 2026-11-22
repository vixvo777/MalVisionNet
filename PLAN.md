# MalVisionNet — Research & Implementation Plan

**Project:** Leakage-aware Windows PE malware detection using binary-to-image learning, lightweight vision models, and optimized CPU inference  
**Document status:** Research-informed implementation specification  
**Primary task:** Binary classification — benign vs. malicious Windows Portable Executable (PE) files  
**Secondary task:** Malware-family recognition and unseen-family evaluation where reliable family labels exist  
**Target implementation:** Python, PyTorch, FastAPI, Next.js (React + TypeScript), ONNX Runtime, Docker Compose  

---

## 0. Executive decision: what we are actually building

### Delivery rule: one final integrated product

This project will have **one complete final application**, not a sequence of partial public releases. Build and verify it phase by phase, but keep all completed components on the same main development line and integrate them into the final system. A phase is complete only when its tests and exit conditions pass. The final handoff happens after the full application, model, API, UI, documentation, and reproducibility checks are complete.

Research checkpoints, experiment IDs, configuration hashes, and model artifact checksums are retained for scientific reproducibility. They do not represent separate product releases.


Build one complete, integrated static-analysis research application that reads Windows PE bytes without executing the file, converts the byte stream into grayscale or three-channel byte images, and evaluates compact CNN/vision-transformer models for benign-versus-malicious classification. The final deliverable includes the audited data pipeline, trained and benchmarked model, optimized ONNX inference, FastAPI backend, and a professional Next.js web application. Development happens in verified phases, but those phases are engineering milestones—not separate public product releases.

The presentation's proposed direction is retained, but several claims are corrected before implementation:

1. **Three-byte RGB mapping is not, by itself, novel.** Nguyen et al. (2023) already compare a three-gram method in which three consecutive bytes form RGB channels, along with grayscale and other image encodings. Their experiments report that the three-gram method is only slightly better than grayscale in their setting; another colour encoding performed better. Therefore, this project must test RGB empirically rather than assume it is superior.
2. **A Vision Transformer is not automatically better or faster.** MobileViT and LeViT are efficiency-oriented vision architectures, but their speed and accuracy on malware byteplots must be measured on our own hardware and test protocol.
3. **A byte-image classifier is not immune to packing or obfuscation.** It avoids disassembly and execution, but packing can still change the image texture and reduce detection quality. We will measure this rather than promise immunity.
4. **A high random-split accuracy is not proof of zero-day detection.** Malware families, file variants, and even different binaries can render to duplicate or near-duplicate images. A recent October 2026 preprint reports substantial metric inflation under leaky image splits and sensitivity to byte-row width. Our primary results must therefore use duplicate-aware, family-aware, and temporal evaluation.
5. **Sub-20 ms latency and 75% model-size reduction are targets to test, not guaranteed outcomes.** We will measure end-to-end CPU latency and quantization effects. We will not publish a target as a result unless the benchmark supports it.

### Working research question

> Under leakage-aware and time-aware evaluation, does a three-byte RGB byteplot combined with a compact vision model offer a useful accuracy–latency trade-off over grayscale CNNs and a conventional static-feature baseline for Windows PE malware detection?

### Intended research contributions

- A reproducible raw-PE-to-byte-image pipeline with grayscale and RGB modes.
- A fair ablation study of byte encoding, byte-row width, image resizing, and compact model families.
- Leakage-aware evaluation using file hashes, rendered-image similarity, malware-family groups, and temporal splits.
- A non-image static-feature baseline evaluated on the same raw-file partitions.
- ONNX CPU inference and a quantified accuracy/latency/model-size trade-off.
- An explainability view that helps an analyst inspect model attention, without presenting saliency as proof of malicious behaviour.

**Do not claim:** a new RGB encoding, guaranteed zero-day detection, complete resistance to obfuscation, production-grade antivirus replacement, or a fixed inference time before measurement.

---

## 1. Scope and threat model

### 1.1 In scope

- Windows PE files: `.exe`, `.dll`, `.sys`, `.scr`, and other validated PE files.
- Static inspection only: read bytes, calculate hashes, parse PE metadata, and create image tensors.
- Binary classification as the primary task.
- Family-held-out experiments as a proxy for generalization to unseen families.
- Model comparison, calibration, error analysis, robustness tests, CPU optimization, and local deployment.

### 1.2 Out of scope for this project

- Running malware, sandbox detonation, dynamic system-call analysis, or network-behaviour collection.
- Claiming true zero-day performance. An unseen-family split is only a controlled proxy.
- Android APK, ELF/Linux, Office macro, script, or memory-dump classification.
- Automatic quarantine/deletion of user files. The completed application returns a prediction and evidence summary only.
- Building a commercial antivirus product.

### 1.3 Threat model and limitations

The attacker may use packing, encryption, padding, section rearrangement, recompilation, or byte-level changes. The classifier sees only a static representation of the file. A model may learn dataset-source artifacts, compiler patterns, file sizes, or repeated family code rather than malicious intent. We will therefore inspect false positives/negatives, use source-aware splits, compare against metadata baselines, and clearly state dataset limitations.

---

## 2. Dataset decision — use these sources, not a random Kaggle dump

The byte-image pipeline needs the **actual binary bytes**. A CSV of extracted PE features or an already-rendered PNG dataset is not a replacement for raw executables.

### 2.1 Primary malware source: BODMAS

**Dataset:** BODMAS — Blue Hexagon Open Dataset for Malware Analysis  
**Official page:** https://whyisyoung.github.io/BODMAS/  
**Paper:** Yang et al., *BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware*, 2021.  

Published dataset facts:

- 57,293 malware samples and 77,142 benign samples in the feature/metadata release.
- Collected from 29 August 2019 to 30 September 2020.
- 581 curated malware families.
- Metadata includes SHA-256, first-seen time, and family label.
- The original PE binaries are released for the malware samples only; original benign binaries are not released.
- Access to the original malware binaries is controlled: the maintainers ask academic users to email from an institutional account, provide an affiliation page, and explain the intended research use. Access is not guaranteed.

**Decision:** Request access to the BODMAS malware binaries using the university email and research affiliation. Use the malware files for the positive class and for family/time-based analysis. Do not assume the BODMAS benign binary files are downloadable; they are not included in the public binary release.

### 2.2 Primary benign source: Assemblage Windows PE

**Dataset:** Assemblage Windows PE corpus  
**Official site:** https://assemblage-dataset.net/  
**Dataset documentation:** https://assemblagedocs.readthedocs.io/en/latest/dataset.html  
**Windows PE Hugging Face dataset:** https://huggingface.co/datasets/changliu8541/Assemblage_PE  
**Paper:** Liu et al., *Assemblage: Automatic Binary Dataset Construction for Machine Learning*, NeurIPS 2024 Datasets and Benchmarks Track.

Published dataset facts:

- The public Windows PE collection is described as approximately 91,000 PE binaries.
- The Hugging Face dataset card reports approximately 43.1 GB total.
- The binaries are built from licensed open-source projects and are accompanied by repository/build provenance and hash metadata.
- Each binary retains the license of its source repository. Check and obey the per-binary license.

**Decision:** Use a carefully selected subset of the Windows PE binaries as the benign corpus. Filter to valid PE files, retain source repository and build provenance, and split by repository/project so near-identical builds from one project cannot cross train/validation/test partitions. The corpus is goodware by construction, but it is not a representative sample of all commercial Windows software; state this limitation.

**Storage warning:** The published archive is large. Before downloading the full corpus, inspect the current dataset card, files, archive size, and license terms. Use a dedicated data drive with sufficient free space. Do not place raw malware binaries in the Git repository.

### 2.3 Optional second benign source: Assemblage vcpkg DLLs

**Dataset:** https://huggingface.co/datasets/changliu8541/Assemblage_vcpkgDLL

This collection contains around 130,000 Windows DLL binaries and is much larger than the initial Windows PE corpus. Use it only as a later external benign-source test or an additional source after the first experiment is stable. It is not required for the first milestone.

### 2.4 Secondary / comparison datasets

| Dataset | What it contains | Correct use in this project | Why it is not the main raw-binary training set |
|---|---|---|---|
| **Malimg** — https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis | 9,458 malware image samples in the original Nataraj paper, across 25 families; the common released image collection is often listed as 9,339 images | Historical reproduction of grayscale malware-family classification and image-pipeline smoke tests | It is an image dataset, not a benign/malware raw-PE corpus; it is old and malware-only |
| **MaleVis** — https://web.cs.hacettepe.edu.tr/~selman/malevis/ | 9,100 training and 5,126 validation RGB images; 25 malware classes plus one legitimate class | Optional image-only external benchmark; use only if the dataset terms permit it | The public data are rendered images; the site says raw binaries require contacting the maintainers |
| **EMBER 2017/2018** — https://github.com/elastic/ember | Large PE feature datasets (not the raw original binaries in the public feature release) | Static-feature/LightGBM benchmark or feature-extraction reference | Cannot independently reproduce the raw-byte-to-image transformation from feature vectors alone |
| **SOREL-20M** — https://github.com/sophos/SOREL-20M | Roughly 10 million benign and 10 million malicious PE metadata/features; a large disarmed malware-binary release | Optional large-scale robustness/transfer experiment if resources and terms allow | Full release is about 8 TB; public malware files are disarmed by zeroing PE `Machine` and `Subsystem` fields; benign binaries are not freely supplied in the public release |
| **Microsoft BIG 2015** — https://www.kaggle.com/competitions/malware-classification | 10,868 labelled malware samples across nine families, with byte dumps and disassembly data | Optional secondary malware-family experiment | Malware-only, very large uncompressed corpus, and the challenge byte representation is not a balanced benign/malware PE dataset |

### 2.5 Dataset choices to avoid for the main byte-image pipeline

- Do not train the byte-image model on EMBER feature vectors as if they were raw file bytes. EMBER is useful for static-feature baselines, not byte-image rendering.
- Do not use only Malimg or MaleVis images and then claim the system has been validated on arbitrary uploaded raw PE files. That would validate an image classifier, not the raw binary acquisition/conversion path.
- Do not treat the 2025 Zenodo *Windows PE Malware Dataset* CSV files as raw executable binaries. Its public file listing contains two CSV files; inspect the actual schema before deciding whether it is useful for feature-level experiments.
- Do not download live samples from arbitrary repositories or run them on the development laptop.

### 2.6 Acquisition gate and practical sample sizes

The project has a **data-access gate** before full training:

1. Request BODMAS malware binary access.
2. Inspect the Assemblage dataset card and confirm storage/licensing requirements.
3. Download only after the access and storage plan is approved.
4. Create a manifest before training.

Recommended scale for the local development machine:

- **Pipeline smoke test:** 100 benign + 100 malware files.
- **First baseline:** 1,000–2,000 per class.
- **First full experiment:** 5,000–10,000 per class, depending on file sizes and available storage.
- **Scale-up:** increase only after training, evaluation, and memory usage are stable.

Do not force equal class counts in the final test set. A balanced subset is acceptable for controlled training experiments, but the final report must also show metrics under a realistic or clearly stated test prevalence.

**Fallback if BODMAS raw-binary access is not granted:** use Malimg/MaleVis only to validate the image model and converter mechanics on existing images; continue requesting a legitimate raw malware research corpus. Do not silently replace the raw-binary task with image-only training and describe it as equivalent.

---

## 3. Literature review and design decisions

The paper list below is the minimum required reading. Read the methods, data section, split strategy, limitations, and metrics — not only the abstract.

| Work | What it contributes | What we take from it / what we must not overclaim |
|---|---|---|
| Nataraj et al. (2011), *Malware Images: Visualization and Automatic Classification*, VizSec | Foundational grayscale byte-image representation; visual texture similarities among malware families | Reproduce grayscale as a baseline. Its high reported family accuracy is not proof of modern benign/malware or unseen-family generalization |
| Kalash et al. (2018), *Malware Classification with Deep Convolutional Neural Networks*, NTMS | CNN-based malware-family image classification | Establish a convolutional image baseline before adding a transformer |
| Raff et al. (2018), *Malware Detection by Eating a Whole EXE*, AAAI Workshop | MalConv processes raw byte sequences directly; demonstrates a non-image deep-learning formulation | Image models must be compared with a non-image baseline where feasible; byte-image resizing loses information differently from sequence models |
| Nguyen, Di Troia, Ishigaki & Stamp (2023), *Generative Adversarial Networks and Image-Based Malware Classification*, Journal of Computer Virology and Hacking Techniques | Compares grayscale, colormap, PE and three-gram image encodings; explicitly tests three consecutive bytes as RGB | RGB/trigram is prior art. Re-test it against grayscale on the same samples and split; do not call it a new encoding |
| Yang et al. (2021), *BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware* | Timestamped PE corpus with 581 curated malware families and concept-drift motivation | Use time and family metadata; test temporal and family generalization rather than random split alone |
| Harang & Rudd (2020), *SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection* | Large-scale PE feature and metadata benchmark with disarmed malware samples | Useful scale reference, but understand disarm modifications and benign-binary access limits |
| Mehta & Rastegari (2022), *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*, ICLR | Lightweight CNN/Transformer hybrid for efficient vision inference | Candidate compact backbone, not a guaranteed winner for byteplots |
| Graham et al. (2021), *LeViT: A Vision Transformer in ConvNet’s Clothing for Faster Inference*, ICCV | Vision transformer designed around speed/accuracy trade-offs | Optional efficient transformer baseline; benchmark on target CPU rather than extrapolating ImageNet results |
| Liu et al. (2024), *Assemblage: Automatic Binary Dataset Construction for Machine Learning*, NeurIPS 2024 | Reproducible source of licensed Windows PE goodware binaries and provenance | Use as the benign corpus and split by source repository/build provenance |
| Alshomrani et al. (2025), *An Explainable Hybrid CNN–Transformer Architecture for Visual Malware Classification*, Sensors | Recent CNN/Transformer fusion and Grad-CAM on multiple image datasets | Hybrid image models and explanations already exist; focus our novelty on raw-PE evaluation, leakage controls, efficiency and ablations |
| Roy & Di Troia (2025), *Discriminative Regions and Adversarial Sensitivity in CNN-Based Malware Image Classification*, Electronics | Studies discriminative image regions and adversarial sensitivity | Include robustness tests; a visually convincing heatmap does not prove semantic explanation or robustness |
| Larson & Leach (2026), *Measuring Duplication, Leakage, and Rendering Dependence in Malware-as-Image Benchmarks*, arXiv preprint | Recent audit of duplication, train/test leakage and rendering-width sensitivity in malware-image benchmarks | Critical warning: exact binary-hash deduplication alone is insufficient; perform rendered-image duplicate checks and width sensitivity tests. Treat this as a recent preprint until independently peer-reviewed |

### 3.1 Research gap we can responsibly investigate

A defensible contribution is **not** “we invented RGB malware images” or “transformers solve malware detection.” The study will test whether RGB byte triplets and lightweight vision backbones retain value when the experimental design prevents image/family/source leakage, and when CPU deployment cost is included.

### 3.2 Hypotheses

- **H1 — Encoding:** RGB byte-triplet input may improve some metrics over grayscale, but the improvement may be small or disappear under strict splits.
- **H2 — Backbone:** MobileViT-XXS/XS or LeViT-128S may provide a competitive accuracy/latency trade-off against ResNet18; this must be observed, not assumed.
- **H3 — Generalization:** Family-held-out and temporal testing will be harder than a random sample split.
- **H4 — Rendering sensitivity:** Image width and resize policy will materially affect performance; the selected rule must be fixed and reported.
- **H5 — Quantization:** ONNX INT8 may reduce model size and/or CPU latency, with a measurable accuracy trade-off.

---

## 4. System architecture

```text
                  RESEARCH DATA PIPELINE
      BODMAS malware binaries + Assemblage PE goodware
                           |
                           v
              SAFE STATIC INGESTION / MANIFEST
        SHA-256 | PE validity | source | date | family
                           |
                           v
               DUPLICATE / LEAKAGE AUDIT
          exact hash + rendered hash + similarity groups
                           |
                           v
                   GROUPED DATA SPLITS
            temporal split + unseen-family split
                           |
               +-----------+-----------+
               |                       |
               v                       v
          GRAYSCALE BYTEPLOT        RGB BYTEPLOT
          1 byte -> 1 pixel         3 bytes -> R,G,B
               |                       |
               +-----------+-----------+
                           v
                 MODEL TRAINING / ABLATION
           LightGBM | ResNet18 | MobileViT | LeViT
                           |
                +----------+----------+
                |                     |
                v                     v
         EVALUATION / XAI        ONNX EXPORT
       F1, PR-AUC, FPR,          FP32 -> INT8
       Grad-CAM, errors               |
                |                     v
                |              CPU BENCHMARK
                |                     |
                +----------+----------+
                           v
               FASTAPI INFERENCE BACKEND
                           |
                    PRIVATE API CALL
                           |
                           v
             NEXT.JS + REACT + TYPESCRIPT UI
       Overview | Analyze | Results | Research metrics
```

### 4.1 Main inference stages

1. Receive file bytes in memory; enforce file-size limits.
2. Calculate SHA-256 and basic file metadata.
3. Validate PE structure with a parser. Do not execute, load, import, or emulate the file.
4. Convert the raw byte stream to the configured grayscale or RGB byteplot.
5. Resize/normalize according to the configuration version used during training.
6. Run the selected model and return probability, thresholded label, model artifact identifier, and processing timings.
7. Display the byteplot and optional saliency map. State clearly that this is a research score, not a guaranteed security verdict.

---

## 5. Byte-to-image conversion specification

The conversion must be implemented once in a tested module and reused by training and inference. Any change to this algorithm creates a new representation configuration identifier and requires a new evaluation.

### 5.1 Grayscale baseline

Given bytes `b[0 ... N-1]`, interpret each byte as an unsigned intensity from 0 to 255 and reshape it into rows of a configured fixed byte width. Pad only the final incomplete row with zeros. Save the original file length and padding count in metadata.

### 5.2 RGB three-byte representation

For the RGB experiment, group consecutive bytes without changing their values:

```python
pixels = bytes_array[:usable_length].reshape(-1, 3)
R = pixels[:, 0]
G = pixels[:, 1]
B = pixels[:, 2]
rgb_pixels = np.stack([R, G, B], axis=1)
```

Conceptually, each pixel is `(byte[i], byte[i+1], byte[i+2])`. This is a three-byte/trigram byteplot. Do not reinterpret it as natural-image colour, and do not silently invert a channel. If a modified mapping is tested, give it a separate experiment name.

### 5.3 Row width and shape policy

- Configure row width in **pixels**, not bytes. For RGB, a width of `W` pixels consumes `3 × W` bytes per row.
- Use one canonical initial configuration (recommended starting point: `W = 256` pixels) for the primary run.
- Run a controlled width sensitivity experiment at `W = 128`, `256`, and `512` pixels. Keep all other settings identical.
- Record the width in every experiment artifact and model metadata.
- Pad only the incomplete final row and record the amount of padding.

### 5.4 Resizing policy

The PDF/presentation proposes fixed-size images (128×128 or 224×224) and bilinear resizing. The research implementation will test the resizing choice rather than silently choose one:

- Primary model input: `224 × 224`.
- Primary downsampling candidate: area-based interpolation (`INTER_AREA`) to reduce aliasing when shrinking a large byteplot.
- Required ablation: bilinear interpolation, because it is part of the proposal.
- Record original file size, source image height/width, resize method, and destination dimensions.
- Do not save PNG/JPEG files during online inference. Convert directly from bytes to a `uint8` NumPy array and then to a tensor. Cache intermediate arrays only during offline training if it measurably improves throughput.

**Important limitation:** Resizing a long byteplot to 224×224 is lossy. It does not preserve every byte as an independent pixel. The report must quantify the effect of resolution and resizing. A later extension may compare multi-window/patch aggregation, but do not add it until the baseline pipeline is correct.

### 5.5 Augmentation rules

Do not apply generic image augmentations blindly. Horizontal/vertical flips, rotations, or aggressive crops change byte order and create patterns that may not correspond to valid executable files. The primary experiment should use no geometric augmentation. Address class imbalance with class-weighted loss or a weighted sampler. Any augmentation experiment must be separately justified and reported.

### 5.6 Converter tests

At minimum test:

- Empty file is rejected with a clear error.
- A known byte sequence maps to the exact expected grayscale values and RGB triplets.
- RGB channel order is correct and no channel is inverted.
- Padding occurs only at the end and is deterministic.
- Width and output dimensions are correct for files smaller/larger than one row.
- Repeated conversion of the same file/configuration yields the same output hash.
- Training and API call the same conversion function and same configuration.
- Input bytes are never executed.

---

## 6. Dataset engineering, provenance, and leakage prevention

This section is a core research component, not optional housekeeping.

### 6.1 Build a manifest before preprocessing

Create one row per source file with at least:

| Field | Meaning |
|---|---|
| `sample_id` | Internal stable identifier; do not use the original filename as a model feature |
| `sha256` | Hash of the exact raw file bytes |
| `label` | `benign` or `malicious` |
| `family` | Malware family if the source provides a defensible label; null for benign/unknown |
| `first_seen` | Source timestamp, if available |
| `source_dataset` | BODMAS, Assemblage, or another documented source |
| `source_project` | Source repository/project for goodware samples |
| `file_size` | Original file size in bytes |
| `pe_valid` | Whether PE validation succeeded |
| `machine` / `subsystem` | Basic PE fields retained for audit only, not automatically fed to the image model |
| `representation_config_id` | Immutable renderer configuration identifier used to create the byteplot |
| `image_hash` | Exact hash of the rendered tensor before normalization |
| `similarity_group` | Group ID for known exact/near duplicates |
| `split` | Train/validation/test assignment |
| `license_or_terms` | Data-use provenance reference |

Never include sample filename, directory name, dataset name, raw SHA-256, or family label as an input feature for the image model.

### 6.2 Validation and filtering

- Verify SHA-256 after download when publisher checksums are available.
- Check PE signatures and file structure. Keep a rejection log; do not silently discard failed files.
- Do not parse executable data by running it.
- Keep malformed/non-PE files in a separate audit table, not in the core PE task.
- Record parser version (e.g. `pefile` or `LIEF`) and version pin.
- Preserve a data dictionary and a dataset card documenting collection, licensing, label source, dates, excluded samples, and known biases.

### 6.3 Duplicate and near-duplicate control

Run these checks before splitting:

1. **Exact binary duplicate:** group identical SHA-256 values; keep one copy or ensure all copies are assigned to the same split.
2. **Exact rendered duplicate:** hash the rendered pixel tensor. Different binaries that render identically must not cross splits.
3. **Near-duplicate images:** compute a documented perceptual hash or similarity embedding on the rendered byteplots; cluster high-similarity pairs and manually inspect a sample of each threshold region.
4. **Binary similarity:** optionally calculate TLSH/ssdeep or another documented fuzzy hash. Treat fuzzy similarity as a grouping/audit signal, not an unquestionable ground-truth family label.
5. **Family/project grouping:** all variants from a held-out malware family must remain out of training in the unseen-family protocol; all builds from a held-out goodware repository must remain out of training in the repository-held-out protocol.

Keep the exact thresholds, library versions, clustering algorithm, and review decisions in `reports/data_audit/`.

### 6.4 Evaluation splits

Create and lock multiple distinct split manifests. Do not tune the model on the final test set.

**Split A — Random/grouped reference split**

- Use only after grouping exact and near-duplicate samples.
- Example proportions: 70% train, 15% validation, 15% test.
- This is a reference result for comparability, not the main generalization claim.

**Split B — Temporal holdout**

- Sort malware samples by BODMAS first-seen timestamp.
- Earlier period: train; following period: validation; latest period: test.
- Use timestamped goodware build/repository metadata where available, and keep goodware projects grouped across splits.
- Report the dates and counts explicitly. Do not invent timestamps for records that have none.

**Split C — Unseen-family holdout**

- Select a group of malware families before model training and exclude those families from train/validation.
- Test the binary classifier on the held-out-family malware together with held-out benign projects.
- The held-out-family task asks whether the model recognizes malicious files from families it did not train on; it is not proof of true zero-day detection.

**Split D — External/source-shift test (optional but valuable)**

- Hold out an entire benign source/project set and, where an additional permitted malware corpus is available, evaluate on that source without retraining.
- If using MaleVis, state that its public data are pre-rendered RGB images. It can validate a separate image-only model, but it cannot validate our raw-PE converter unless the maintainers provide the original binaries and the terms allow use.

### 6.5 Source leakage and class confounding

Because BODMAS malware and Assemblage benign binaries come from different collection processes, the model may learn source/collection artifacts. This is a major limitation. Mitigate and investigate it by:

- keeping multiple benign repositories/projects represented across splits;
- splitting benign files by source repository, never by individual file alone;
- reporting source-specific performance and error rates;
- extracting static PE metadata from the exact same raw files and training a LightGBM baseline on the same split;
- performing feature ablations for file size, timestamp, section counts, signatures, and other metadata to see whether simple provenance features dominate;
- clearly stating that the assembled corpus is a research benchmark, not a perfectly representative sample of all Windows endpoint traffic.

---

## 7. Models and baselines

Do not build the largest model first. Train a small, reproducible baseline and only add complexity when it produces a measurable benefit.

### 7.1 Required baselines

1. **Static-feature LightGBM:** extract a consistent PE feature set (using the same parser version) from the exact raw samples in the benchmark. This is the non-image baseline.
2. **Grayscale + ResNet18:** baseline visual model following the established grayscale byteplot family of methods.
3. **RGB-trigram + ResNet18:** isolates the effect of RGB encoding while keeping the backbone fixed.
4. **RGB-trigram + MobileViT-XXS or XS:** primary proposed model.

### 7.2 Optional model

- **RGB-trigram + LeViT-128S:** include only if the implementation, export, and training are stable within the available compute budget.
- **Raw-byte sequence model (MalConv-style):** a valuable future comparison, but it is not a first-milestone requirement because raw sequence models have different input-length and memory challenges.

### 7.3 Fair comparison rules

- The same file IDs and locked split manifests must be used for all models.
- Grayscale and RGB models must see the same underlying files.
- Use the same label definition and report class prevalence.
- Tune only on validation data.
- Run at least three random seeds for the final main comparison, if compute allows.
- Report parameter count, checkpoint size, CPU inference time, and preprocessing time in addition to predictive metrics.

### 7.4 Initial training configuration

Starting values only; tune on validation data and log all changes.

| Setting | Starting value |
|---|---|
| Input | 224×224, 1-channel grayscale or 3-channel RGB |
| Optimizer | AdamW |
| Learning rate | 3e-4 for a new classifier head; lower fine-tuning LR (e.g. 1e-5) for pretrained backbone |
| Weight decay | 1e-4 to 1e-3 search range |
| Batch size | Start at 8; increase to 16 only if GPU memory permits |
| Epochs | 10–30 with early stopping |
| Early stopping | Validation PR-AUC or macro-F1; patience 5 |
| Loss | Binary cross-entropy with logits; class weights based on training partition only |
| Precision | Mixed precision on CUDA if numerically stable |
| Seeds | 3 for final experiments |
| Checkpoint selection | Best validation metric, never test metric |

With a 6 GB laptop GPU, begin with small batches, mixed precision, and gradient accumulation if needed. Do not assume the full dataset will fit in GPU RAM. Stream or batch samples and profile host RAM as well.

### 7.5 Explainability

- Use Grad-CAM or an architecture-compatible saliency method for CNN-based models.
- For transformer-based models, use a documented attention rollout or gradient-based explanation method supported by the chosen implementation.
- Display heatmaps beside the original byteplot, with a note that the highlighted pixels indicate model sensitivity, not a validated malicious code region.
- Inspect examples where the heatmap focuses on padding, file boundaries, or a repeated dataset artifact. These may indicate shortcut learning.

---

## 8. Experiment matrix

Complete these in order. Do not start every experiment at once.

| ID | Question | Change | Keep fixed | Primary outputs |
|---|---|---|---|---|
| E0 | Is the data pipeline correct? | 100+100 sample smoke test | Tiny fixed manifest | Converter tests, PE validation, duplicate report |
| E1 | What is the baseline? | Grayscale + ResNet18 | Fixed split and 224×224 | Precision, recall, macro-F1, PR-AUC, ROC-AUC |
| E2 | Does RGB help? | RGB + ResNet18 vs grayscale + ResNet18 | Same files, split, training budget | Metric deltas and confidence intervals |
| E3 | Does MobileViT help? | RGB + MobileViT vs RGB + ResNet18 | Same samples and representation | Accuracy/latency/model-size trade-off |
| E4 | Is row width driving results? | 128 vs 256 vs 512 pixel widths | Same model and files | Width sensitivity under grouped/temporal split |
| E5 | Does resize interpolation matter? | Area vs bilinear | Same width/model/data | Metric and stability comparison |
| E6 | Does it generalize? | Random vs temporal vs unseen-family splits | Same model/config | Performance degradation and error analysis |
| E7 | Can we optimize inference? | PyTorch FP32 vs ONNX FP32 vs ONNX INT8 | Same model and test files | Accuracy, file size, RAM, p50/p95 latency |
| E8 | Is it interpretable/useful? | Grad-CAM/attention views and error review | Selected checkpoint | Correct/incorrect examples, shortcut indicators |

### 8.1 Required result tables

**Predictive performance:** Accuracy, balanced accuracy, precision, recall, macro-F1, PR-AUC, ROC-AUC, and confusion matrix. Report malicious recall and benign false-positive rate separately.

**Operational performance:** Model size, peak CPU RAM, preprocessing latency, model-only latency, end-to-end latency, p50 and p95, and throughput. Record hardware, thread count, batch size, runtime version, and warm-up protocol.

**Calibration:** Brier score or expected calibration error, reliability plot, and chosen decision threshold. The threshold must be selected using the validation set according to the stated false-positive/recall trade-off.

**Robustness:** Performance by family, file-size bucket, packed/high-entropy subgroup if metadata supports it, PE validity subgroup, and source dataset. Do not label a sample “packed” from entropy alone without stating that this is a heuristic.

---

## 9. Evaluation protocol — metrics that matter

Accuracy alone is not enough for malware detection, especially with class imbalance.

- **Precision:** how many files flagged as malicious are actually labelled malicious.
- **Recall / TPR:** how much labelled malware the detector catches.
- **False-positive rate (FPR):** how often benign files are incorrectly flagged.
- **Macro-F1:** balances class-level F1 rather than allowing the majority class to dominate.
- **PR-AUC:** useful when malware prevalence is low or class imbalance is material.
- **ROC-AUC:** overall ranking quality across thresholds; do not treat it as an operational threshold result.
- **FPR at a fixed recall:** report, for example, FPR at 95% malware recall if the validation/test sample size supports a reliable estimate.
- **Calibration:** assess whether 0.9 confidence corresponds to approximately 90% correctness.
- **Latency:** include the entire path from bytes received to score returned, not just neural-network forward time.

For key metrics, use bootstrap confidence intervals over the test set. Keep sample-level predictions so results can be reproduced and errors inspected.

### 9.1 What counts as a successful research result

The project succeeds if it delivers a leakage-audited dataset manifest, a reproducible conversion pipeline, fair baseline/model comparisons, strict generalization evaluations, and measured CPU deployment results. The RGB/transformer model does **not** need to beat every baseline for the study to be valuable; a well-supported negative result is still research.

Do not set “95%+ F1”, “zero-day detection”, “sub-20 ms”, or “75% smaller” as pass/fail claims before results exist. Report measured values with the exact test protocol and hardware.

---

## 10. ONNX and CPU optimization

### 10.1 Export sequence

1. Select the best checkpoint using validation data.
2. Export to ONNX FP32 with a fixed input signature and supported opset.
3. Compare PyTorch FP32 and ONNX FP32 probabilities on a fixed validation batch.
4. Establish an acceptable numerical tolerance and document any model-specific differences.
5. Benchmark ONNX FP32 on CPU.
6. Quantize and benchmark again.
7. Re-evaluate predictive metrics after quantization; do not assume quantization preserves the decision boundary.

### 10.2 Quantization strategy

- For CNN-heavy models, start with **static quantization** using representative calibration samples. ONNX Runtime documentation generally recommends static quantization for CNNs.
- For transformer/linear-heavy models, test **dynamic quantization** where operators and backend support it. ONNX Runtime documentation generally recommends dynamic quantization for transformers and RNNs.
- If post-training quantization causes unacceptable performance loss, investigate quantization-aware training later.
- Keep FP32 model, ONNX FP32 model, and INT8 model as separately named, checksummed artifacts. These are research/build artifacts, not separate product releases.

### 10.3 Benchmark rules

- Use the same machine, CPU provider, thread count, and test files for each variant.
- Warm up before measuring.
- Measure at least 500–1,000 repeated inferences where feasible; report p50 and p95.
- Report image conversion time separately from model time, then report total end-to-end latency.
- Include cold-start model loading separately from per-file inference.
- Never claim sub-20 ms from the network forward pass alone while excluding parsing, hashing, conversion, and API overhead.

---

## 11. Backend and professional web application

The final application uses a clear separation of responsibilities:

- **FastAPI** owns file validation, preprocessing, model inference, result schemas, and health/model metadata.
- **Next.js (React + TypeScript)** owns the complete user interface and server-side API proxying.
- **ONNX Runtime** runs the selected optimized model on CPU. A PyTorch inference path remains available for controlled parity checks during development.
- **Docker Compose** starts the final web and API services together on a private internal network.


### 11.1 FastAPI service

Required endpoints:

- `GET /health` — service readiness and model-loaded status.
- `GET /model-info` — model name, model artifact identifier, representation configuration identifier, decision threshold, input shape, and runtime provider.
- `POST /predict` — accepts one validated PE file and returns prediction, probability, SHA-256, safe file metadata, model artifact identifier, representation configuration identifier, and a timing breakdown.
- `GET /research/metrics` — returns the final approved aggregate benchmark metrics generated by the evaluation pipeline. It must not trigger training.

Safety and API requirements:

- Enforce file-size, request-time, and concurrency limits.
- Validate PE format and reject unsupported or malformed files cleanly.
- Process bytes in memory; do not persist uploaded binaries by default.
- Never execute, load, import, or emulate the submitted file.
- Do not send file bytes to VirusTotal or another third-party service.
- Avoid logging file contents or sensitive user data. Log only request ID, safe metadata, status, and timings needed for debugging.
- Return a research disclaimer and explain that model confidence is not proof of safety.
- Keep CORS restrictive. The browser must not receive unrestricted access to the private inference service.

### 11.2 Next.js web application

**Frontend stack**

- Next.js using the App Router.
- React + TypeScript with strict type checking.
- Tailwind CSS for styling.
- shadcn/ui (Radix UI primitives) for accessible interface components.
- Recharts for benchmark charts where charts are needed.
- Zod for runtime validation of API responses and form inputs.

**Final application screens**

1. **Overview (`/`)** — project summary, model readiness, high-level research metrics, supported file types, and a clear research-use disclaimer.
2. **Analyze (`/analyze`)** — drag-and-drop or file-picker upload, PE validation feedback, safe metadata preview, scan progress, and result state. The page must clearly state that uploaded files are read statically and never executed.
3. **Analysis result (shown after upload on `/analyze`)** — benign/malicious score, calibrated confidence where available, configured decision threshold, SHA-256, file size, PE metadata, byteplot visualization, timing breakdown, and explanation overlay when supported. Do not describe a saliency map as proof of malicious intent. The result is held for the current analysis session; raw uploaded files are not stored.
4. **Research (`/research`)** — final model comparison, confusion matrix, precision/recall/F1, PR and ROC curves, calibration, unseen-family and temporal results, and CPU latency/size comparisons. Clearly label the evaluation split used for every metric.
5. **System information** — model artifact identifier, representation configuration, runtime provider, API health, and limitations.

**Frontend engineering rules**

- Use reusable typed components: `FileDropzone`, `ScanProgress`, `VerdictCard`, `ConfidenceMeter`, `ByteplotViewer`, `MetadataTable`, `TimingBreakdown`, `MetricCard`, and `BenchmarkChart`.
- Keep upload, loading, success, malformed-file, unsupported-file, API-error, and oversized-file states explicit. Never leave the page in an ambiguous spinner state.
- Use a Next.js server-side route handler as a backend-for-frontend (BFF) proxy for upload and API calls. The browser calls the Next.js application; the Next.js server forwards the request to FastAPI over the private Docker network.
- Do not embed secrets, filesystem paths, or private API addresses in client-side bundles.
- Apply accessible labels, keyboard navigation, focus states, responsive layouts, and readable contrast.
- Do not display fake benchmark values. The Research page reads only the final metrics artifact produced by the evaluation pipeline.
- Do not provide a public malware download, sharing, or execution feature.

### 11.3 Final application integration and containerization

Use Docker Compose to run the Next.js web service and FastAPI inference service together. Keep the API on an internal network where practical; expose only the web entry point to the local host. Run containers as non-root users, restrict filesystem access, avoid privileged mode and Docker socket mounts, and do not provide any execution route for submitted binaries. Keep raw datasets outside images/containers and outside source control.

The final handoff is one integrated application with a single documented startup path. Internal phases may have test builds and checkpoints, but no partial phase is treated as a public product release.
---

## 12. Recommended repository structure

```text
MalVisionNet/
├── PLAN.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── representation.yaml
│   ├── model.yaml
│   └── experiments/
│       ├── e1_gray_resnet18.yaml
│       ├── e2_rgb_resnet18.yaml
│       ├── e3_rgb_mobilevit.yaml
│       └── e4_width_sensitivity.yaml
├── data/                         # never commit raw data
│   ├── raw/
│   │   ├── bodmas/
│   │   └── assemblage/
│   ├── manifests/
│   ├── interim/
│   └── processed/
├── src/
│   └── malvision/
│       ├── data/
│       │   ├── build_manifest.py
│       │   ├── validate_pe.py
│       │   ├── deduplicate.py
│       │   └── make_splits.py
│       ├── representations/
│       │   ├── grayscale.py
│       │   ├── rgb_trigram.py
│       │   └── resize.py
│       ├── features/
│       │   └── extract_pe_features.py
│       ├── models/
│       │   ├── resnet18.py
│       │   ├── mobilevit.py
│       │   └── levit.py
│       ├── training/
│       │   ├── train.py
│       │   ├── losses.py
│       │   └── metrics.py
│       ├── evaluation/
│       │   ├── evaluate.py
│       │   ├── robustness.py
│       │   └── error_analysis.py
│       ├── explainability/
│       │   └── saliency.py
│       ├── export/
│       │   ├── export_onnx.py
│       │   ├── quantize.py
│       │   └── benchmark.py
│       ├── api/
│       │   └── main.py
├── frontend/                    # Next.js web application
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── app/
│       │   ├── page.tsx
│       │   ├── layout.tsx
│       │   ├── analyze/page.tsx
│       │   ├── research/page.tsx
│       │   └── api/
│       │       ├── analyze/route.ts
│       │       ├── health/route.ts
│       │       └── research/metrics/route.ts
│       ├── components/
│       │   ├── navigation/
│       │   ├── scanner/
│       │   ├── results/
│       │   └── charts/
│       ├── lib/
│       │   ├── api.ts
│       │   └── schemas.ts
│       └── types/
├── tests/
│   ├── test_byteplot.py
│   ├── test_pe_validation.py
│   ├── test_splits.py
│   └── test_api.py
├── artifacts/                    # checkpoints and ONNX files; do not commit large files
│   ├── checkpoints/
│   ├── onnx/
│   └── reports/
├── reports/
│   ├── data_audit/
│   ├── experiments/
│   └── figures/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   └── 02_error_analysis.ipynb
└── docker-compose.yml
```

### 12.1 Git and data policy

Add at minimum:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.DS_Store
.vscode/
.idea/

# Raw or derived datasets — never push malware binaries or dataset archives
/data/raw/
/data/interim/
/data/processed/
*.exe
*.dll
*.sys
*.scr
*.bytes
*.7z
*.rar
*.zip
*.tar
*.tar.gz
*.tar.xz

# Model artifacts and generated outputs
/artifacts/checkpoints/
/artifacts/onnx/
*.pt
*.pth
*.onnx
*.safetensors
/mlruns/
wandb/

# Next.js frontend build outputs
/frontend/node_modules/
/frontend/.next/
/frontend/out/
/frontend/coverage/
```

Commit source code, configs, tests, data manifests (only if license/terms permit), experiment metadata, and aggregate results. Do not commit raw malware or restricted data. Do not upload restricted binaries to cloud storage or public repositories.

---

## 13. Development environment

### 13.1 Recommended environment

- Python 3.11 (pin exact patch/minor in the project environment file).
- PyTorch + torchvision installed using the official selector for the local NVIDIA driver/CUDA combination.
- `timm` for supported MobileViT/LeViT implementations.
- `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `opencv-python-headless`, `Pillow`, `tqdm`.
- `pefile` and/or `lief` for PE parsing; pin parser version for feature consistency.
- `onnx`, `onnxruntime`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `mlflow`.
- Optional: `imagehash` and a TLSH/ssdeep implementation for duplicate auditing. Pin and document their versions.

Install one OpenCV package only (`opencv-python-headless` is appropriate for this server/data pipeline; do not also install `opencv-python`). Install Node.js LTS for the Next.js frontend; use the package manager and lockfile selected for the repository.

### 13.2 Windows PowerShell setup

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

Install PyTorch and torchvision from the official PyTorch installation selector for the machine's driver/CUDA version. Then install the remaining packages:

```powershell
pip install numpy pandas scikit-learn lightgbm opencv-python-headless pillow tqdm timm pefile lief
pip install onnx onnxruntime fastapi uvicorn pydantic mlflow pytest imagehash
```

Verify the environment:

```powershell
python -c "import sys, torch; print(sys.version); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Save the exact working package versions after the first successful environment setup:

```powershell
pip freeze > requirements-lock.txt
```

Do not blindly copy a CUDA wheel command from an old tutorial. The correct PyTorch wheel depends on the current driver and PyTorch release.

---

## 14. Implementation sequence — phase by phase

### Phase 0 — Research/data access and project skeleton

**Tasks**

- Read the required literature in Section 3.
- Email BODMAS maintainers for research access to the original malware binaries.
- Inspect Assemblage Windows PE dataset card, archive layout, provenance, and licenses.
- Create repository, environment, `.gitignore`, config system, and experiment naming convention.
- Create a data-use register and storage plan.

**Deliverables**

- `PLAN.md`, `README.md`, environment lockfile, dataset access status, initial data dictionary.
- No model training yet.

**Gate:** Do not proceed to live malware binary acquisition until the access terms and isolated storage/processing plan are clear.

### Phase 1 — Data audit and byte-image engine

**Tasks**

- Build the manifest from allowed sample metadata.
- Validate PE format and compute SHA-256.
- Log malformed files and duplicate groups.
- Implement grayscale and RGB-trigram converters.
- Implement fixed-width and resizing options.
- Add unit tests for converter invariants.
- Render a small sample grid for manual QA.

**Deliverables**

- `reports/data_audit/` with class/source/date/file-size counts and duplicate analysis.
- Passing converter and manifest tests.
- Versioned conversion config.

**Gate:** No training until the converter, split, and duplicate checks pass.

### Phase 2 — Baselines

**Tasks**

- Generate locked grouped and temporal split manifests.
- Train the LightGBM static-feature baseline on the same raw-file partitions.
- Train grayscale + ResNet18.
- Save per-sample predictions and confusion matrices.
- Check model behaviour on a small curated false-positive/false-negative list.

**Deliverables**

- Reproducible baseline checkpoints and metrics.
- First written result summary.

### Phase 3 — RGB and compact vision models

**Tasks**

- Train RGB-trigram + ResNet18.
- Train RGB-trigram + MobileViT-XXS/XS.
- Add LeViT-128S only if the first two models are stable.
- Run width and resize ablations.
- Compare results under random/grouped, temporal, and unseen-family protocols.

**Deliverables**

- Main experiment table and confidence intervals.
- Analysis of whether RGB and/or the transformer really help.

### Phase 4 — Robustness and explainability

**Tasks**

- Evaluate by family, file size, source, and available packing/high-entropy labels.
- Run controlled byte-row width and resize sensitivity tests.
- Inspect duplicate clusters and possible source leakage.
- Generate Grad-CAM/attention views for true positives, false positives, false negatives, and true negatives.
- Review whether explanations focus on padding, headers, or suspicious repeated patterns.

**Deliverables**

- Robustness report, error taxonomy, explanation gallery.

### Phase 5 — Export and optimization

**Tasks**

- Export best model to ONNX FP32.
- Verify PyTorch/ONNX prediction parity.
- Test supported static/dynamic INT8 quantization paths.
- Benchmark CPU p50/p95 latency, model size, RAM, and end-to-end request latency.
- Recalculate all main metrics for the quantized model.

**Deliverables**

- FP32 and INT8 ONNX artifacts, parity tests, benchmark CSV and plots.

### Phase 6 — FastAPI inference backend

**Tasks**

- Implement the FastAPI health, model-info, predict, and research-metrics endpoints.
- Add PE validation, upload limits, in-memory processing, response schemas, and structured safe logging.
- Integrate the selected ONNX model and verify output parity with the approved PyTorch checkpoint.
- Add API tests for valid PE files, malformed files, non-PE files, oversized uploads, empty files, and unavailable model artifacts.

**Deliverables**

- Tested FastAPI inference service, OpenAPI documentation, and repeatable local API startup instructions.

### Phase 7 — Next.js analyst web application

**Tasks**

- Build the Next.js App Router application using React, TypeScript, Tailwind CSS, and accessible UI components.
- Implement Overview, Analyze, Analysis Result, Research, and System Information screens.
- Create the server-side BFF route handlers that forward requests to FastAPI without exposing private backend configuration to the browser.
- Implement drag-and-drop and file-picker upload, explicit loading/error states, byteplot display, metrics charts, and responsive layouts.
- Connect the Research page only to metrics generated by the locked evaluation pipeline; do not use placeholder or invented values.
- Add frontend type checks, linting, component tests, and end-to-end tests for the upload-to-result journey.

**Deliverables**

- Complete Next.js frontend integrated with the inference API, with all final screens and tested user journeys.

### Phase 8 — Final integration, security and release hardening

**Tasks**

- Build the web and API services with Docker Compose and verify private service-to-service communication.
- Test the full upload-to-result path on supported Windows PE samples and all rejection/error cases.
- Verify the application never executes uploaded bytes and does not persist raw files by default.
- Run a clean-machine reproducibility check using documented environment and startup commands.
- Freeze the final model artifact, representation configuration, dataset/split manifests, metrics, and container build configuration.
- Fix all release-blocking defects. Do not publish an incomplete feature set as an interim product.

**Deliverables**

- One complete, integrated final application; Docker Compose deployment; security/test reports; final model and configuration checksums; and a single documented startup procedure.

### Phase 9 — Final research report and presentation

**Tasks**

- Freeze final split, configuration, source versions, and model checkpoint.
- Re-run the final experiment from a clean environment.
- Write the methodology, dataset card, metrics, limitations, and reproducibility appendix.
- Include negative results and failed hypotheses.
- Prepare an honest demonstration with benign test files and safe static metadata.

**Deliverables**

- Final report/paper draft, slides, complete GitHub repository, model card, dataset card, benchmark artifacts, and the single integrated final application.

---

## 15. Quality checklist — no shortcuts

### Dataset

- [ ] All raw files have provenance and permitted-use notes.
- [ ] BODMAS binary access and conditions are documented.
- [ ] Assemblage repository/build identifiers are retained.
- [ ] SHA-256 computed and verified.
- [ ] Exact binary and exact rendered-image duplicates grouped.
- [ ] Near-duplicate threshold and manual audit documented.
- [ ] No raw malware binaries committed to Git or uploaded publicly.
- [ ] No sample executed during ingestion, preprocessing, training, testing, or API inference.

### Representation

- [ ] RGB means consecutive byte triplets with explicit R/G/B order.
- [ ] Grayscale baseline works.
- [ ] Width is measured in pixels and the exact configuration is immutable and checksummed.
- [ ] Resize method and original file dimensions are recorded.
- [ ] Converter tests pass on edge cases.
- [ ] Same code path used in training and inference.

### Evaluation

- [ ] Grouped split prevents exact/near-duplicate leakage.
- [ ] Temporal split is locked and dates are reported.
- [ ] Unseen-family split is independent from the training families.
- [ ] Repository grouping is used for benign Assemblage data.
- [ ] All models use the same benchmark samples and splits.
- [ ] Accuracy is not the only reported metric.
- [ ] Threshold is selected on validation data only.
- [ ] Per-sample predictions are saved for error analysis.
- [ ] Results include confidence intervals or explain why not.

### Optimization and deployment

- [ ] ONNX outputs are checked against PyTorch outputs.
- [ ] INT8 accuracy is re-evaluated.
- [ ] End-to-end CPU latency is measured, not guessed.
- [ ] API rejects malformed/oversized input safely.
- [ ] API never executes submitted bytes.
- [ ] Model artifact and representation configuration identifiers are returned with each prediction.
- [ ] Web application communicates uncertainty and limitations.

---

## 16. Main risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| BODMAS binary access is not granted | The main raw malware source is permission-based | Apply early; keep a clearly labelled image-only development fallback; do not misrepresent it as raw-PE validation |
| Benign/malware source mismatch | Model may learn corpus origin rather than maliciousness | Repository-group splits, multiple benign sources, source-specific metrics, same-file static-feature baseline, cautious claims |
| Near-duplicate leakage | Inflates test metrics | SHA-256 + rendered-tensor hashes + perceptual/fuzzy duplicate grouping before split |
| Byte-row width sensitivity | The rendering rule itself can drive model scores | Freeze a canonical width and report 128/256/512 sensitivity |
| RGB does not improve performance | The hypothesis may be false | Report the negative result; use grayscale if it is more reliable or faster |
| Transformer export unsupported/slow | Some model operations may not export or quantize cleanly | Start with ResNet18 and MobileViT; keep LeViT optional; test ONNX export early |
| INT8 harms recall | Quantization can change probabilities and thresholds | Re-evaluate all metrics; recalibrate threshold on validation; keep FP32 fallback |
| Limited storage/RAM/VRAM | Archives and image caches are large | Start with a small subset, stream files, use separate data drive, cache only what is useful, small batches |
| Live malware exposure | Raw samples may be dangerous if mishandled | Separate isolated research VM/storage, no execution, no network sharing, no personal daily-use folder, no public upload |
| Misleading “zero-day” claim | Unseen family is not identical to a genuine new threat | Say “unseen-family evaluation” and describe the limits |

---

## 17. Minimum reproducibility artifacts

Every experiment must save:

- Git commit hash.
- Configuration YAML and representation configuration identifier.
- Dataset manifest version/hash and split manifest hash.
- Random seed, library versions, GPU/CPU information.
- Model checkpoint and best epoch.
- Training log and validation curves.
- Test predictions with sample IDs (not raw malware bytes).
- Classification report, confusion matrix, ROC/PR curves, calibration plot.
- CPU benchmark output with software/hardware details.
- A short experiment note that explains what changed from the previous run.

Use MLflow locally to track parameters, metrics, artifacts, and run IDs. Do not log raw malware files as MLflow artifacts.

---

## 18. Suggested project schedule

A realistic research-first schedule is approximately **8–9 weeks**, depending mainly on BODMAS data access, download time, and model-training duration. The phases are sequential engineering gates; they do not represent separate user-facing releases.

| Week | Focus | Exit condition |
|---|---|---|
| 1 | Literature, access requests, repository/environment, dataset plan | Data permissions, isolation and storage plan documented |
| 2 | Manifest, PE audit, duplicate groups, converters and tests | Deterministic converters and audited data manifest |
| 3 | Grouped splits, LightGBM and grayscale ResNet baseline | Reproducible baselines and error records |
| 4 | RGB ResNet and MobileViT training | Fair encoding/backbone comparison |
| 5 | Width/resizing, temporal/family/source tests | Generalization and representation sensitivity results |
| 6 | Explainability, robustness and final model selection | Selected model supported by error and robustness analysis |
| 7 | ONNX, INT8, CPU profiling and FastAPI | Measured inference backend with parity tests |
| 8 | Complete Next.js web application and API integration | All final user journeys work end to end |
| 9 | Docker hardening, clean rerun, final report and slides | One complete final application and reproducible research handoff |

If BODMAS access is delayed, Weeks 2–5 can use Malimg/MaleVis only for image-model pipeline development, but the raw-PE research claims remain pending until a permitted raw malware-binary corpus is available. The final application is not declared complete until the raw-PE evaluation requirement is met or the research scope is explicitly revised and disclosed.

---

## 19. First actions to start the project

Do these in order:

1. Create the repository and save this file as root `PLAN.md`.
2. Send the BODMAS dataset-access request from the university email, including a concise defensive research description and the institutional affiliation page requested by the maintainers.
3. Open the Assemblage Windows PE dataset card and inspect `binaries.csv`, archive layout, and per-repository license data. Estimate local disk needs before downloading.
4. Create the Python 3.11 environment and run the hardware verification command.
5. Build the manifest + converter on a tiny benign-only test sample first. Do not acquire or process raw malware until the isolated handling setup is ready.
6. Add the unit tests before starting any model training.
7. Run the grayscale baseline before RGB/transformer experiments. Do not jump straight to MobileViT or ONNX.

### Definition of done

The project is complete when another student can reproduce the dataset manifest and split decisions (without receiving restricted raw binaries), run the conversion tests, train and evaluate the baselines, reproduce the main result tables, verify ONNX parity, and launch the complete local Next.js + FastAPI application. The report must clearly disclose all dataset access limits, source mismatch, duplicate-control procedure, and the difference between unseen-family testing and true zero-day detection.

---

## 20. References and source links

1. L. Nataraj, S. Karthikeyan, G. Jacob, and B. S. Manjunath, “Malware Images: Visualization and Automatic Classification,” *Proceedings of the 8th International Symposium on Visualization for Cyber Security (VizSec)*, 2011, pp. 1–7. DOI: https://doi.org/10.1145/2016904.2016908. PDF: https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf
2. M. Kalash et al., “Malware Classification with Deep Convolutional Neural Networks,” *9th IFIP International Conference on New Technologies, Mobility and Security (NTMS)*, 2018. https://doi.org/10.1109/NTMS.2018.8328749
3. E. Raff, J. Barker, J. Sylvester, R. Brandon, B. Catanzaro, and C. Nicholas, “Malware Detection by Eating a Whole EXE,” *AAAI Workshop on Artificial Intelligence for Cyber Security*, 2018. https://cdn.aaai.org/ocs/ws/ws0432/16422-75958-1-PB.pdf
4. H. Nguyen, F. Di Troia, G. Ishigaki, and M. Stamp, “Generative Adversarial Networks and Image-Based Malware Classification,” *Journal of Computer Virology and Hacking Techniques*, 19, 579–595, 2023. https://doi.org/10.1007/s11416-023-00465-2
5. L. Yang, A. Ciptadi, I. Laziuk, A. Ahmadzadeh, and G. Wang, “BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware,” *4th Deep Learning and Security Workshop*, 2021. DOI: https://doi.org/10.1109/SPW53761.2021.00020. Dataset: https://whyisyoung.github.io/BODMAS/
6. R. Harang and E. M. Rudd, “SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection,” arXiv:2012.07634, 2020. https://arxiv.org/abs/2012.07634
7. S. Mehta and M. Rastegari, “MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer,” ICLR, 2022. https://arxiv.org/abs/2110.02178
8. B. Graham et al., “LeViT: A Vision Transformer in ConvNet’s Clothing for Faster Inference,” ICCV, 2021, pp. 12259–12269. https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html
9. C. Liu et al., “Assemblage: Automatic Binary Dataset Construction for Machine Learning,” *NeurIPS 2024 Datasets and Benchmarks Track*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bbefc73a187dd42e0dc065b4e7a0615-Abstract.html
10. M. Alshomrani, A. Albeshri, A. A. Alsulami, and B. Alturki, “An Explainable Hybrid CNN–Transformer Architecture for Visual Malware Classification,” *Sensors*, 25(15), 4581, 2025. https://doi.org/10.3390/s25154581
11. A. Roy and F. Di Troia, “Discriminative Regions and Adversarial Sensitivity in CNN-Based Malware Image Classification,” *Electronics*, 14(19), 3937, 2025. https://doi.org/10.3390/electronics14193937
12. S. Larson and K. Leach, “Measuring Duplication, Leakage, and Rendering Dependence in Malware-as-Image Benchmarks,” arXiv:2610.39351, 29 October 2026. https://arxiv.org/abs/2610.39351 — recent preprint; check for a peer-reviewed version before final publication.
13. ONNX Runtime, “Quantize ONNX models.” https://onnxruntime.ai/docs/how-to/quantization.html
14. Elastic, “EMBER: Elastic Malware Benchmark for Empowering Researchers.” https://github.com/elastic/ember
15. MaleVis dataset page, “MaleVis: A Dataset for Vision Based Malware Recognition.” https://web.cs.hacettepe.edu.tr/~selman/malevis/
16. Assemblage dataset documentation. https://assemblagedocs.readthedocs.io/en/latest/dataset.html

---

**Final research principle:** Build the most defensible experiment, not the most impressive-sounding architecture. A carefully audited result showing when RGB, MobileViT, or INT8 helps — and when it does not — is stronger research than an unsupported claim of 99% accuracy or “zero-day protection.”
