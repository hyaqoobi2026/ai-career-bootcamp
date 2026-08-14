"""Generate the 16-week day-by-day curriculum as curriculum.json.

Schedule model (from the agreed 40 hrs/week split):
  Mon-Sat : 3h morning DEEP block (implementation + math)
            3h evening BUILD block (project / course / reading)
            ~1.5h job hunt
  Sunday  : lighter day - 4h flex (review, catch-up, weekly retro) + job hunt

Edit the PHASES/WEEKS structures below and re-run:  python3 build_curriculum.py
"""
from __future__ import annotations

import datetime as dt
import json

START = dt.date(2026, 8, 30)          # Sunday
WEEKS = 16

# --------------------------------------------------------------------------- phases
PHASES = [
    {"n": 1, "name": "Foundations", "weeks": [1, 2, 3, 4],
     "goal": "Write every core algorithm by hand. Backprop stops being abstract.",
     "color": "#4f7cff"},
    {"n": 2, "name": "Deep Learning Core", "weeks": [5, 6, 7, 8],
     "goal": "PyTorch fluency, real training runs, your first models on real data.",
     "color": "#7c5cff"},
    {"n": 3, "name": "Transformers & LLMs", "weeks": [9, 10, 11, 12],
     "goal": "Build a GPT from scratch. Understand attention at the code level.",
     "color": "#b84fd6"},
    {"n": 4, "name": "The Capstone Experiment", "weeks": [13, 14, 15, 16],
     "goal": "Fine-tune on your own labelled data, benchmark vs the LLM cascade, publish it.",
     "color": "#e0574f"},
]

# --------------------------------------------------------------------------- weekly plan
# Each week: theme + the 6 working days' deep/build tasks. Sunday is generated.
W = {}

W[1] = {"theme": "Gradient descent by hand",
        "deep": [
            ("Linear regression from scratch", "numpy only. Generate y=3x+2+noise. Derive dW and db ON PAPER first, then code the loop. Print loss every 100 steps. Success = w≈3, b≈2.", 180),
            ("Linear regression, multivariate", "Extend to several features. Vectorise with matrix ops. Add a learning-rate experiment: try 0.001, 0.01, 0.1 — watch what diverges.", 180),
            ("Logistic regression from scratch", "Sigmoid + binary cross-entropy, gradients by hand. Train on a toy 2-class dataset. Plot the decision boundary.", 180),
            ("Linear algebra I", "3Blue1Brown 'Essence of Linear Algebra' ch 1-8. For each video, write one numpy example proving what it showed.", 180),
            ("Linear algebra II", "Vectors, matrices as transformations, dot product, matrix multiply — implement matmul yourself in pure Python, then compare to numpy.", 180),
            ("Week-1 rebuild", "From a blank file, rewrite linear + logistic regression with no reference. This is the retention test.", 180),
        ],
        "build": [
            ("Environment setup", "Python 3.11+, venv, numpy/matplotlib/pandas, VS Code, Jupyter. Create a git repo 'ai-lab' and commit day 1.", 90),
            ("Hands-On ML ch.1", "Géron ch.1 — the ML landscape. Take notes in your repo as markdown.", 90),
            ("Kaggle account + GPU check", "Sign up, run a notebook, confirm your 30 free GPU hrs/week. Run nvidia-smi.", 60),
            ("Hands-On ML ch.2", "End-to-end project chapter. Follow along in your own notebook — do not copy-paste.", 120),
            ("Plot your training curves", "matplotlib: loss over steps for all 3 models you built. Save as PNGs in the repo.", 90),
            ("Write-up: what I learned", "One markdown page in your repo explaining gradient descent as if to a colleague. Teaching = the real test.", 90),
        ]}

W[2] = {"theme": "Neural network from scratch",
        "deep": [
            ("Neuron + forward pass", "One neuron, then a layer. Implement forward pass with numpy. Understand shapes obsessively — print every shape.", 180),
            ("Manual backprop, 1 hidden layer", "Derive the chain rule on paper for a 2-layer net. Then code it. This is the hardest day of the month — expect struggle.", 180),
            ("Train it on XOR", "Prove your net learns a non-linear function. If XOR works, your backprop is correct.", 180),
            ("Activation functions", "Implement sigmoid, tanh, ReLU + their derivatives. Compare training curves on the same problem.", 180),
            ("MNIST with your own net", "Load MNIST, train your from-scratch net. Target >90% accuracy. No frameworks.", 180),
            ("Week-2 rebuild", "Blank file. Rebuild the 2-layer net + backprop from memory.", 180),
        ],
        "build": [
            ("Calculus refresher", "Khan/MIT 18.02: partial derivatives, chain rule, gradients. Do 10 problems by hand.", 120),
            ("Hands-On ML ch.4", "Training models — gradient descent variants, regularisation.", 120),
            ("Matrix calculus notes", "Write your own cheat-sheet: dL/dW shapes for a linear layer. You will use this forever.", 90),
            ("Debug clinic", "Deliberately break your net (wrong shape, wrong sign) and practise reading the error. Debugging is the skill.", 90),
            ("Vectorisation", "Rewrite your loops as matrix ops. Time both. Feel the difference.", 90),
            ("Write-up: backprop explained", "Markdown page deriving backprop for your 2-layer net.", 90),
        ]}

W[3] = {"theme": "micrograd — build an autograd engine",
        "deep": [
            ("Karpathy micrograd pt.1", "Watch + build along: the Value class, forward ops, the graph.", 180),
            ("micrograd pt.2 — backward", "Implement _backward for +, *, tanh. Topological sort. This is the core of PyTorch in 100 lines.", 180),
            ("micrograd pt.3 — an MLP", "Build Neuron/Layer/MLP on your engine. Train it on a toy dataset.", 180),
            ("Extend micrograd", "Add exp, log, ReLU, and division yourself — not from the video. Verify gradients numerically.", 180),
            ("Gradient checking", "Implement numerical gradient checking to prove your analytic grads are right. This is a real research skill.", 180),
            ("Week-3 rebuild", "Rebuild micrograd from a blank file. Aim for under 150 lines.", 180),
        ],
        "build": [
            ("PyTorch intro", "Install torch. Tensors, autograd, .backward(). Re-do week-2's net in PyTorch — feel what the framework removed.", 120),
            ("Compare: yours vs torch", "Same problem, your engine vs PyTorch. Compare gradients numerically. They should match.", 120),
            ("Hands-On ML ch.10", "Intro to ANNs with Keras — skim for concepts, you are using torch.", 90),
            ("Repo hygiene", "README with what you built, requirements.txt, clean commit history. This repo becomes portfolio.", 90),
            ("Read: Backprop paper", "Rumelhart, Hinton & Williams (1986). Read the original. It is short and readable.", 90),
            ("Write-up: micrograd", "Explain your autograd engine in a markdown post. Candidate for your first blog post.", 90),
        ]}

W[4] = {"theme": "Classical ML + first gate",
        "deep": [
            ("k-NN + k-means from scratch", "Both in numpy. Understand distance metrics and the curse of dimensionality.", 180),
            ("Decision tree from scratch", "Gini/entropy, recursive splitting. Then understand why boosting works.", 180),
            ("PCA from scratch", "Covariance matrix, eigendecomposition, projection. Ties directly to your linear algebra.", 180),
            ("scikit-learn sprint", "Now use the library: LogisticRegression, RandomForest, GradientBoosting on a tabular dataset. Compare to your implementations.", 180),
            ("Evaluation deep-dive", "Precision, recall, F1, ROC-AUC, confusion matrix, class imbalance. Implement each metric yourself.", 180),
            ("GATE 1", "Assessment: from blank files, in 3 hours — linear regression, logistic regression, and a 2-layer net with backprop. No references.", 180),
        ],
        "build": [
            ("Hands-On ML ch.6-7", "Decision trees and ensembles.", 120),
            ("First Kaggle entry", "Titanic or House Prices. Submit something. The point is finishing, not the score.", 150),
            ("Cross-validation", "Implement k-fold yourself, then use sklearn's. Understand why leakage matters.", 90),
            ("AHRC data exploration", "Load your incident data. Profile it: class balance, missing fields, text lengths. This is your capstone dataset.", 120),
            ("Read: ESL ch.2", "Elements of Statistical Learning — overview of supervised learning (free PDF).", 90),
            ("Month-1 retro", "What worked, what slipped, what to change. Write it down.", 60),
        ]}

W[5] = {"theme": "PyTorch properly + probability",
        "deep": [
            ("PyTorch fundamentals", "Datasets, DataLoaders, nn.Module, optimisers, schedulers. Build the training loop you will reuse forever.", 180),
            ("Probability I — Stat 110", "Blitzstein lectures 1-4: sample spaces, conditional probability, Bayes. Do the problem sets.", 180),
            ("Training loop patterns", "Train/val split, early stopping, checkpointing, seeding for reproducibility. Make it a reusable template.", 180),
            ("Probability II", "Stat 110 lec 5-9: random variables, expectation, variance, common distributions.", 180),
            ("Regularisation lab", "Dropout, weight decay, batch norm. Run controlled experiments and plot the differences.", 180),
            ("Week rebuild + review", "Rebuild the training template from memory. Review the week's probability.", 180),
        ],
        "build": [
            ("Hands-On ML ch.11", "Training deep nets — vanishing gradients, initialisation, optimisers.", 120),
            ("Experiment tracking", "Set up simple logging (CSV or Weights & Biases free tier). Track every run from now on.", 90),
            ("Kaggle GPU run", "Move a training job to Kaggle GPU. Learn the workflow — you will need it for transformers.", 120),
            ("AHRC baseline", "TF-IDF + logistic regression on your incident classification. This is the baseline every later model must beat.", 150),
            ("Read: Adam paper", "Kingma & Ba (2014). You are using it — understand it.", 90),
            ("Write-up: baseline results", "Document the baseline with numbers. First entry in your capstone log.", 60),
        ]}

W[6] = {"theme": "Convolutional networks",
        "deep": [
            ("Convolution from scratch", "Implement 2D convolution in numpy. Understand kernels, stride, padding.", 180),
            ("CNN in PyTorch", "Build a small CNN. Train on CIFAR-10 from scratch on Kaggle GPU.", 180),
            ("Probability III", "Stat 110 lec 10-14: joint distributions, covariance, LLN, CLT.", 180),
            ("Data augmentation", "Implement augmentation, measure the accuracy delta. Learn to prove an improvement.", 180),
            ("Transfer learning", "Fine-tune a pretrained ResNet. Compare to training from scratch — note the data-efficiency gap.", 180),
            ("Week rebuild", "CNN from blank file. Then review probability.", 180),
        ],
        "build": [
            ("Hands-On ML ch.14", "Deep computer vision with CNNs.", 120),
            ("Read: ResNet paper", "He et al. (2015). Residual connections — one of the most important ideas in DL.", 90),
            ("Ablation practice", "Take your CNN, remove one component at a time, measure. This is how research is actually done.", 120),
            ("AHRC: text features", "Explore embeddings vs TF-IDF on your data. Measure retrieval quality.", 120),
            ("Portfolio site", "Start documenting projects on mhadiyaqoobi.com. One page per project.", 120),
            ("Weekly retro", "Log hours actually done vs planned. Adjust.", 60),
        ]}

W[7] = {"theme": "Sequences — RNN to attention",
        "deep": [
            ("makemore pt.1", "Karpathy: bigram language model. Build it, sample from it.", 180),
            ("makemore pt.2 — MLP", "Bengio-style neural LM. Embeddings appear for the first time.", 180),
            ("makemore pt.3", "Batch norm, initialisation, diagnostics. Watch activation statistics — this is real practitioner skill.", 180),
            ("RNN from scratch", "Implement a vanilla RNN and train it. Feel the vanishing-gradient problem yourself.", 180),
            ("Probability IV", "Stat 110: conditional expectation, Markov chains. Directly relevant to language models.", 180),
            ("Week rebuild", "Rebuild the bigram + MLP language model from memory.", 180),
        ],
        "build": [
            ("Read: word2vec", "Mikolov et al. Embeddings from first principles.", 90),
            ("Tokenisation", "Implement BPE yourself. Then use HuggingFace tokenizers. Critical for your multilingual work.", 150),
            ("Dari/Pashto tokenisation", "Test tokenisers on Perso-Arabic script. Measure token counts vs English — you will find real problems.", 120),
            ("HF NLP course ch.1-2", "Hugging Face course — free, directly job-relevant.", 120),
            ("AHRC: language stats", "Analyse your corpus by language. Document what is under-represented.", 90),
            ("Weekly retro", "Review + plan.", 60),
        ]}

W[8] = {"theme": "Consolidation + Gate 2",
        "deep": [
            ("Attention from scratch", "Implement scaled dot-product attention in numpy. By hand, small matrices, printed shapes.", 180),
            ("Multi-head attention", "Extend to multi-head in PyTorch. Understand why multiple heads help.", 180),
            ("Positional encoding", "Implement sinusoidal and learned. Understand why order needs encoding at all.", 180),
            ("Optimisation theory", "Convex vs non-convex, SGD/momentum/Adam derivations. Boyd ch.1-3 skim.", 180),
            ("Review week", "Re-read your own write-ups. Rebuild anything shaky.", 180),
            ("GATE 2", "Assessment: implement attention from scratch + train a CNN to >70% on CIFAR-10, in 3 hours, no references.", 180),
        ],
        "build": [
            ("Read: Attention Is All You Need", "Vaswani et al. (2017). Read it properly, twice. Annotate.", 150),
            ("The Illustrated Transformer", "Jay Alammar's post. The best visual explanation there is.", 90),
            ("Blog post #1", "Publish 'Building micrograd: how backprop actually works'. Post it on LinkedIn.", 180),
            ("AHRC: label audit", "Extract your human-reviewed labels into a clean dataset. Document class balance and quality.", 120),
            ("Month-2 retro", "Honest assessment. Adjust the plan for phase 3.", 60),
            ("Rest + review", "Lighter build day. Review notes.", 60),
        ]}

W[9] = {"theme": "nanoGPT — build a transformer",
        "deep": [
            ("Karpathy GPT pt.1", "'Let's build GPT from scratch' — follow along, type every line.", 180),
            ("GPT pt.2 — self-attention", "The attention block, masking, causal attention. The heart of it.", 180),
            ("GPT pt.3 — blocks", "Residuals, layer norm, feed-forward. Assemble the full transformer block.", 180),
            ("GPT pt.4 — train it", "Train on tiny Shakespeare. Watch it learn to produce English-ish text.", 180),
            ("Rebuild nanoGPT", "From a blank file, no video. This is the single most valuable rebuild in the programme.", 180),
            ("Scaling experiments", "Vary layers, heads, embedding size. Plot loss vs parameters. See scaling laws yourself.", 180),
        ],
        "build": [
            ("Read: GPT-2 paper", "Radford et al. Language models as multi-task learners.", 120),
            ("Read: scaling laws", "Kaplan et al. (2020) — why bigger works.", 120),
            ("HF Transformers", "Load a pretrained model, generate text, inspect internals.", 120),
            ("Dari/Pashto corpus", "Build a clean text corpus from your AHRC sources. Document size and preprocessing.", 150),
            ("Portfolio update", "Add nanoGPT to your site with your scaling plots.", 90),
            ("Weekly retro", "Review + plan.", 60),
        ]}

W[10] = {"theme": "Your own GPT + embeddings",
        "deep": [
            ("Train GPT on your corpus", "Train nanoGPT on Dari/Pashto text. Almost nobody has done this. Document everything.", 180),
            ("Tokeniser experiments", "Train a custom BPE tokeniser on Perso-Arabic. Compare compression vs the multilingual defaults.", 180),
            ("Embeddings from scratch", "Implement skip-gram word2vec. Train on your corpus. Inspect nearest neighbours.", 180),
            ("Sentence embeddings", "Use sentence-transformers. Build semantic search over your incidents. Compare to your feature-hashing embedder.", 180),
            ("Evaluate retrieval", "Build a small labelled retrieval set. Measure recall@k. Prove the improvement with numbers.", 180),
            ("Week rebuild + review", "Rebuild the training pipeline from memory.", 180),
        ],
        "build": [
            ("Read: BGE-M3 / multilingual embeddings", "How modern multilingual embedding models are trained.", 120),
            ("Vector search", "FAISS or sqlite-vec. Build a working semantic search demo.", 150),
            ("HF NLP course ch.5-6", "Datasets and tokenisers.", 120),
            ("AHRC: swap the embedder", "Replace feature-hashing with real embeddings in your platform. Measure the retrieval delta.", 150),
            ("Blog post #2", "'Training a language model on a low-resource language' — this will get attention.", 150),
            ("Weekly retro", "Review + plan.", 60),
        ]}

W[11] = {"theme": "Fine-tuning",
        "deep": [
            ("Fine-tuning fundamentals", "Full fine-tune a small BERT on a public classification task. Learn the workflow end to end.", 180),
            ("LoRA / PEFT", "Parameter-efficient fine-tuning. Implement LoRA conceptually, then use the library.", 180),
            ("Instruction tuning", "How chat models are made. Read InstructGPT, understand SFT vs RLHF vs DPO.", 180),
            ("Quantisation", "4-bit/8-bit. Take a model, shrink it, measure the accuracy/latency tradeoff.", 180),
            ("Distillation", "Train a small model from a large model's outputs. This is directly your capstone technique.", 180),
            ("Week rebuild", "Fine-tuning pipeline from memory.", 180),
        ],
        "build": [
            ("Read: LoRA paper", "Hu et al. (2021).", 90),
            ("Read: InstructGPT", "Ouyang et al. (2022).", 120),
            ("ModernBERT", "Read about it, load it, test on your task.", 120),
            ("AHRC: dataset prep", "Build train/val/test splits from your labelled incidents. Guard against leakage.", 150),
            ("Compute plan", "Work out exactly what you can train on Kaggle's 30 free hrs. Plan the capstone runs.", 90),
            ("Weekly retro", "Review + plan.", 60),
        ]}

W[12] = {"theme": "Evaluation + Gate 3",
        "deep": [
            ("Evaluation methodology", "Design the capstone evaluation: metrics, splits, baselines, statistical significance.", 180),
            ("Calibration", "Implement reliability diagrams and ECE. Apply to your own models. This is your differentiator.", 180),
            ("Error analysis", "Systematic error analysis on your baseline. Categorise every failure. This is what separates practitioners.", 180),
            ("LLM evaluation", "How to evaluate generative systems: groundedness, LLM-as-judge and its failure modes.", 180),
            ("Review week", "Consolidate phases 1-3. Rebuild anything weak.", 180),
            ("GATE 3", "Assessment: build + train a transformer from scratch and fine-tune a pretrained model, 3 hours, no references.", 180),
        ],
        "build": [
            ("Read: eval papers", "HELM and a recent LLM-eval survey. Note what nobody has solved.", 120),
            ("Build an eval harness", "Reusable evaluation code for the capstone. Reproducible, seeded, logged.", 150),
            ("Blog post #3", "'How I evaluate AI systems in high-stakes settings'. Your niche, staked out publicly.", 150),
            ("Month-3 retro", "Honest review. You are 3/4 through — assess the plan.", 90),
            ("Capstone plan", "Write the full experiment plan: hypothesis, method, metrics, deliverables.", 120),
            ("Rest + review", "Light day.", 60),
        ]}

W[13] = {"theme": "CAPSTONE: build the pipeline",
        "deep": [
            ("Capstone: data pipeline", "Clean, split, and version your AHRC labelled dataset. Reproducible from a script.", 180),
            ("Capstone: baselines", "Re-run TF-IDF+LR and the current LLM cascade on the frozen test set. Record cost and latency.", 180),
            ("Capstone: fine-tune v1", "First fine-tuning run on ModernBERT or similar. Log everything.", 180),
            ("Capstone: iterate", "Hyperparameter sweep on Kaggle GPU. Track every run.", 180),
            ("Capstone: multilingual", "Test on Dari/Pashto subsets specifically. This is where your work is novel.", 180),
            ("Week review", "Assess results so far. Adjust method.", 180),
        ],
        "build": [
            ("Experiment log", "Structured log of every run: config, metrics, cost. This becomes the paper's results table.", 120),
            ("Read: related work", "Find 10 papers on low-resource text classification. Build your bibliography.", 150),
            ("Cost analysis", "Compute cost-per-item for every approach. This is the headline result for industry.", 120),
            ("Job hunt push", "Dedicated block: applications, follow-ups, networking.", 120),
            ("Portfolio update", "Publish capstone progress.", 90),
            ("Weekly retro", "Review.", 60),
        ]}

W[14] = {"theme": "CAPSTONE: results",
        "deep": [
            ("Few-shot experiments", "How well can you do with 50, 100, 200 labels? The data-efficiency curve is a key finding.", 180),
            ("Synthetic labels", "Use an LLM to generate training data. Measure whether it helps. Honest either way.", 180),
            ("Distillation run", "Distil your LLM cascade into a small model. The core experiment.", 180),
            ("Error analysis", "Categorise every failure of the best model. Find the pattern.", 180),
            ("Calibration + thresholds", "Calibrate the final model. Recommend an evidence-based confidence threshold.", 180),
            ("Results consolidation", "Final results table. Every number reproducible.", 180),
        ],
        "build": [
            ("Figures", "Publication-quality plots: cost vs accuracy, data-efficiency curve, reliability diagram.", 150),
            ("Reproducibility", "Anyone should be able to re-run this from your repo. Seeds, configs, README.", 120),
            ("Read: writing guides", "How to write an ML paper. Structure before prose.", 90),
            ("Job hunt push", "Applications + interviews.", 120),
            ("Draft: abstract + intro", "Start the write-up while results are fresh.", 150),
            ("Weekly retro", "Review.", 60),
        ]}

W[15] = {"theme": "CAPSTONE: publish",
        "deep": [
            ("Write: method section", "Precise enough to reproduce.", 180),
            ("Write: results section", "Tables, figures, honest limitations.", 180),
            ("Write: discussion", "What it means, what it does not, what is next.", 180),
            ("Write: related work", "Position against the 10 papers you read.", 180),
            ("Full draft", "Assemble and edit. Read aloud.", 180),
            ("Revise", "Second pass. Cut 20% of the words.", 180),
        ],
        "build": [
            ("Blog version", "A readable version for LinkedIn/your site. Lead with the cost number.", 150),
            ("Open-source the code", "Clean repo, MIT licence, real README with results.", 150),
            ("Identify venues", "ACL workshops, NLP4PI, COMPASS. Note deadlines.", 120),
            ("Find collaborators", "Contact Cohere For AI / Aya, ML Collective. Offer your Dari/Pashto expertise.", 120),
            ("Job hunt push", "Applications + interviews.", 120),
            ("Weekly retro", "Review.", 60),
        ]}

W[16] = {"theme": "Ship + Gate 4",
        "deep": [
            ("Deploy the model", "Serve the fine-tuned model behind an API. Measure latency.", 180),
            ("Integrate into AHRC", "Wire it into the real platform behind a flag. Compare live against the cascade.", 180),
            ("Monitoring", "Drift detection and logging for the deployed model.", 180),
            ("Documentation", "Model card: intended use, limitations, evaluation, ethical considerations.", 180),
            ("Review the whole 16 weeks", "Re-read every write-up. Note what to deepen next.", 180),
            ("GATE 4", "Final assessment: full interview simulation — 2 technical problems, one system design, one project deep-dive.", 180),
        ],
        "build": [
            ("Publish everything", "Blog post live, repo public, paper on arXiv or submitted.", 150),
            ("Update LinkedIn + CV", "New skills, new project, new results. Rewrite the headline with real numbers.", 120),
            ("Announce it", "LinkedIn post with the headline result. Tag relevant communities.", 90),
            ("Plan phase 2 (months 5-12)", "Based on what you learned, plan the next block.", 150),
            ("Job hunt push", "Interviews with a much stronger portfolio.", 120),
            ("Celebrate + rest", "You built and shipped a real ML system in 16 weeks. Take the day.", 60),
        ]}

# ~6 hrs/day behind the wheel is already spent. It costs nothing extra to fill it.
# One audio item per week, listened to across the week's drives.
AUDIO = {
    1: ("Lex Fridman / Karpathy interviews", "Start with Karpathy on Lex. Then Hinton, then Sutskever. Free on YouTube/Spotify — you are building the map of who thinks what."),
    2: ("Your own AHRC interview pack", "Upload ~/Desktop/AHRC-interview-prep/*.md to NotebookLM and generate audio. Listening to your own system explained back to you is the fastest way to become fluent in it."),
    3: ("3Blue1Brown neural network series (audio)", "You will have watched these. Re-listen while driving — repetition is what moves it into long-term memory."),
    4: ("Practical AI / TWIML podcast", "Pick episodes on production ML. You are learning the vocabulary practitioners use."),
    5: ("Stat 110 lectures (audio)", "Blitzstein's lectures work surprisingly well as audio. Re-listen to what you studied that morning."),
    6: ("Hands-On ML chapters via text-to-speech", "Push the chapter into a TTS app. Skim first at the desk, then listen to consolidate."),
    7: ("Karpathy makemore series", "Re-listen to the lecture you worked through that morning."),
    8: ("Attention Is All You Need — NotebookLM audio", "Upload the paper to NotebookLM. Listen until the architecture is obvious to you."),
    9: ("GPT / scaling-laws papers via NotebookLM", "Turn the week's papers into an audio discussion and listen on repeat."),
    10: ("Multilingual NLP talks", "Search for Aya / Cohere For AI talks on low-resource languages. This is your niche — learn who is in it."),
    11: ("LoRA + fine-tuning explainers", "Podcasts and conference talks on practical fine-tuning."),
    12: ("Evaluation + AI safety talks", "Anthropic and DeepMind talks on evaluation. Directly relevant to what you want to be known for."),
    13: ("Your own capstone notes via NotebookLM", "Upload your experiment log. Listening to your own results forces you to notice the gaps."),
    14: ("Interview prep — your story bank", "NotebookLM audio of 11_War_Stories_and_Tradeoffs.md. Rehearse answers out loud in the car."),
    15: ("Paper-writing and research talks", "How researchers structure and pitch work."),
    16: ("Career + negotiation", "Salary negotiation talks. You will need this sooner than you think."),
}

JOB_TASKS = [
    ("Applications", "Apply to 5 roles (BSA / Data Analyst / Solutions Architect). Quality over spray — tailor the first line.", 90),
    ("Follow-ups + agencies", "Chase pending applications. Contact one staffing agency (TEKsystems, Insight Global, Robert Half, Motion).", 90),
    ("Networking", "Two messages: a former colleague or a new connection in your target space.", 60),
    ("Applications", "Apply to 5 roles.", 90),
    ("Interview prep", "Rehearse your project story out loud. Time it: 2 min and 10 min versions.", 90),
    ("Applications", "Apply to 5 roles. Log every one in your tracker.", 90),
]

# Day 1 falls on a Sunday. A "weekly review" on day one reviews nothing —
# open with orientation instead.
DAY_ONE = [
    ("Read the whole plan", "Page through all 16 weeks in this app. Know where you are going before you start walking.", 45),
    ("Set up the machine", "Python 3.11+, venv, numpy/pandas/matplotlib, VS Code, Jupyter. Verify each import runs.", 90),
    ("Create the lab repo", "New GitHub repo 'ai-lab'. README with your goal and start date. First commit today.", 45),
    ("Write your why", "One page, in the repo: why you are doing this and what you want in 12 months. You will re-read this in week 9 when it gets hard.", 30),
    ("Block the time", "Put the morning and evening blocks in your phone calendar for all 16 weeks, repeating. Decide your wake-up time.", 40),
    ("Job hunt setup", "Build your application tracker (spreadsheet: role, company, date, status, follow-up). This runs the whole 16 weeks.", 50),
]

SUNDAY = [
    ("Weekly review", "What got done vs planned? What slipped and why? Write 5 lines in your log.", 60),
    ("Catch-up / flex", "Finish whatever slipped this week. If nothing slipped, go deeper on the hardest topic.", 120),
    ("Plan the week ahead", "Read next week's tasks. Block the time in your calendar. Prepare any downloads/datasets.", 60),
    ("Job hunt admin", "Update your application tracker. Prepare for any interviews next week.", 60),
]


def phase_for(week: int) -> dict:
    return next(p for p in PHASES if week in p["weeks"])


def build() -> dict:
    days = []
    for i in range(WEEKS * 7):
        date = START + dt.timedelta(days=i)
        week = i // 7 + 1
        dow = i % 7                       # 0 = Sunday (START is a Sunday)
        ph = phase_for(week)
        spec = W[week]
        tasks = []

        if i == 0:                        # day one — orientation
            for t, (title, detail, mins) in enumerate(DAY_ONE):
                tasks.append({"id": f"d0-o{t}", "block": "Day one — set up",
                              "cat": "job" if "Job hunt" in title else "review",
                              "title": title, "detail": detail, "mins": mins})
        elif dow == 0:                    # Sunday — lighter day
            for t, (title, detail, mins) in enumerate(SUNDAY):
                tasks.append({"id": f"d{i}-s{t}", "block": "Flex",
                              "cat": "review", "title": title,
                              "detail": detail, "mins": mins})
        else:
            k = dow - 1                   # 0..5 across Mon-Sat
            title, detail, mins = spec["deep"][k]
            tasks.append({"id": f"d{i}-deep", "block": "Morning — deep work",
                          "cat": "implementation" if "algebra" not in title.lower()
                          and "probability" not in title.lower()
                          and "calculus" not in title.lower()
                          and "optimisation" not in title.lower() else "math",
                          "title": title, "detail": detail, "mins": mins})
            title, detail, mins = spec["build"][k]
            cat = ("reading" if title.lower().startswith("read")
                   else "project" if "ahrc" in title.lower() or "capstone" in title.lower()
                   else "course")
            tasks.append({"id": f"d{i}-build", "block": "Evening — build & learn",
                          "cat": cat, "title": title, "detail": detail, "mins": mins})
            jt, jd, jm = JOB_TASKS[k]
            tasks.append({"id": f"d{i}-job", "block": "Job hunt",
                          "cat": "job", "title": jt, "detail": jd, "mins": jm})

        # every day, including day one and Sundays — the car time is already spent
        at, ad = AUDIO[week]
        tasks.append({"id": f"d{i}-aud", "block": "While driving",
                      "cat": "audio", "title": at, "detail": ad, "mins": 300})

        days.append({
            "date": date.isoformat(),
            "dow": date.strftime("%a"),
            "week": week,
            "phase": ph["n"],
            "phaseName": ph["name"],
            "phaseColor": ph["color"],
            "theme": "Start here" if i == 0 else spec["theme"],
            "isGate": any("GATE" in t["title"] for t in tasks),
            "tasks": tasks,
        })

    return {
        "meta": {
            "title": "AI Career Bootcamp",
            "start": START.isoformat(),
            "weeks": WEEKS,
            "generated": "build_curriculum.py",
            "owner": "M Hadi Yaqoobi",
        },
        "phases": PHASES,
        "days": days,
    }


if __name__ == "__main__":
    data = build()
    with open("curriculum.json", "w") as f:
        json.dump(data, f, indent=1)
    total = sum(t["mins"] for d in data["days"] for t in d["tasks"])
    print(f"days: {len(data['days'])}  tasks: {sum(len(d['tasks']) for d in data['days'])}")
    print(f"total planned: {total/60:.0f} h  (~{total/60/WEEKS:.1f} h/week)")
