# Assignment 3: Multi-Dimensional Classification of Extremist Content

**Course:** 04637-1: Mobile Big Data Analytics and Management
**Final Deadline:** March 25, 2026

---

## Team Members

| Name              |
|-------------------|
| Athanase Bahizire |
| Ukachi Eze-Mbey   |
| Joe Karangwa      |
| Catherine Makori  |
| Arsene Mugisha    |

---

## Table of Contents

1. [Dataset](#dataset)
2. [Deliverables](#deliverables)
3. [Part A: Multi-Dimensional Extremism Classification](#part-a-multi-dimensional-extremism-classification)
   - [A1. Key Differences and Similarities Between VE and NVE](#a1-key-differences-and-similarities-between-ve-and-nve)
   - [A2. How Radicalization and Extremism Are Defined in the Method Section](#a2-how-radicalization-and-extremism-are-defined-in-the-method-section)
   - [A3. Definitions of VE and NVE and How They Shaped Our Prompt](#a3-definitions-of-ve-and-nve-and-how-they-shaped-our-prompt)
   - [A4. VE Subtype Classification](#a4-ve-subtype-classification)
   - [A5. Target Group Identification](#a5-target-group-identification)
   - [A6. Prompt Design Choices](#a6-prompt-design-choices)
   - [A7. Classification Results and Analysis](#a7-classification-results-and-analysis)
   - [A8. Interpretability and Explanation](#a8-interpretability-and-explanation)
4. [Part B: Prompt Engineering Comparison](#part-b-prompt-engineering-comparison)
   - [B1. Severity Levels](#b1-severity-levels)
   - [B2. Sample Extraction](#b2-sample-extraction)
   - [B3. Prompt Design: Zero-Shot vs Chain-of-Thought](#b3-prompt-design-zero-shot-vs-chain-of-thought)
   - [B4. Experiment Conditions](#b4-experiment-conditions)
   - [B5. Comparison Analysis](#b5-comparison-analysis)
5. [Part C: Counter-Narrative Generation](#part-c-counter-narrative-generation)
   - [C1. Persona-Adaptive Counter-Narratives](#c1-persona-adaptive-counter-narratives)
   - [C2. LLM Comparison](#c2-llm-comparison)
   - [C3. Verbosity and Readability Analysis](#c3-verbosity-and-readability-analysis)
6. [Model Selection Notes](#model-selection-notes)

---

## Dataset

**Source:** [Digital Extremism Detection – Curated Dataset (Kaggle)](https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset/)

The dataset contains social media posts labelled as `EXTREMIST` or `NON_EXTREMIST`. A balanced sample of 150 examples (75 from each class) was drawn using `random_state=42` for reproducibility before running the multi-dimensional classification pipeline.

---

## Deliverables

| Deliverable | Status |
|---|---|
| Notebook (`cmakori_LLM_configuration_Lab.ipynb`) | Included |
| `ve_nve_classifications.csv` | Included |
| README (this file) | Included |
| Peer group evaluation | Included separately |

---

## Part A: Multi-Dimensional Extremism Classification

**Model used:** `claude-sonnet-4-20250514-v1:0` (substituted for `gpt-4o-mini-2024-07-18` due to CMU AI Gateway authentication failures — see [Model Selection Notes](#model-selection-notes))

---

### A1. Key Differences and Similarities Between VE and NVE

**Based on:** Knight, Woodward, & Lancaster (2017), *Violent Versus Nonviolent Actors: An Empirical Study of Different Types of Extremism*

#### Similarities

Knight et al. (2017) conducted in-depth case studies on 40 extremist individuals and found a striking number of shared characteristics between Violent Extremists (VEs) and Non-Violent Extremists (NVEs). Both groups universally exhibited a perceived external threat or outgroup to blame (100% in both VEs and NVEs). Both groups showed strong evidence of seeking like-minded others and of being driven by a clear ideology (100% in both groups). A shared sense of purpose and meaning, identity-seeking behaviour, and outgroup hostility were also equally prevalent across the two categories. The authors note that these commonly cited radicalization indicators — such as dehumanization of others, perceived persecution, and identification with persecuted communities — appear at comparable rates in both VEs and NVEs and therefore cannot alone predict whether an individual will turn to violence.

#### Differences

Despite these shared foundations, Knight et al. (2017) identified several statistically significant variables that distinguish VEs from NVEs. The most decisive distinguishing variable was a personal sense of responsibility to act, which was present in 87.5% of VEs but only 18.8% of NVEs. This sense of obligation to take direct action appears to be a critical factor in the transition from ideological commitment to physical violence.

Additional distinguishing characteristics of VEs (relative to NVEs) include:

- **Exposure to extreme violence** (87.5% VE vs. 56.3% NVE) — including through online materials
- **Experiences of bullying** (45.8% VE vs. 6.3% NVE)
- **Low self-esteem** (33.3% VE vs. 6.3% NVE)
- **Deliberate disconnection from certain social groups** (54.2% VE vs. 12.5% NVE)
- **Underachievement** — a perceived gap between academic potential and employment outcomes (70.8% VE vs. 31.3% NVE)
- **Participation in extremist-related training** (45.8% VE vs. 0% NVE)
- **Travel abroad for extremist purposes** (45.8% VE vs. 12.5% NVE)
- **Passionate team sport membership** (54.2% VE vs. 12.5% NVE) — associated with a sense of belonging, discipline, and group identity

VEs were also more likely to be operating in an open environment with few security constraints, giving them greater physical opportunity to act on their beliefs. This "open operating environment" (91.7% VE vs. 43.8% NVE) is a particularly practical indicator for counter-terrorism risk assessment.

The authors also acknowledge an important methodological caution: some individuals categorized as NVEs may be better described as "pre-violent" — people who had not yet acted violently but could be on a pathway toward it, having been disrupted before that transition occurred.

---

### A2. How Radicalization and Extremism Are Defined in the Method Section

In the Method section of Knight et al. (2017), the authors devote considerable attention to the definitional inconsistency that plagues the existing literature. They note that many studies fail to define extremism clearly, conflate extremist attitudes with extremist actions, or use radicalization and extremism interchangeably to describe both thought processes and violent outcomes.

#### Definition of Radicalization

The authors draw on the UK Home Office definition, which frames radicalization as "the process by which people come to support terrorism and extremism and, in some cases, to then participate in terrorist activity." However, the authors are careful to distance their own study from any deterministic view of radicalization. They explicitly state that their study does not examine how people become radicalized — instead, the focus is on how, when, and why individuals choose different extremist-related actions, whether violent or nonviolent. They further note that radicalization is only one of various pathways into violent extremism, and that radical opinion is not a necessary condition for radical action (citing McCauley & Moskalenko, 2014). Radicalization is therefore treated as contextual background rather than as the primary object of study.

#### Definition of Extremism

The UK Government's working definition of extremism — used as the reference point in the study — defines it as the "vocal or active opposition to fundamental British values, including democracy, the rule of law, individual liberty and mutual respect and tolerance of different faiths and beliefs." The authors apply this pragmatically: in their study, extremists are individuals known to hold attitudes and beliefs that diverge from mainstream political or ideological opinion, and who had been convicted of a criminal offense where there was evidence that the offense was motivated by extremism. This conviction-grounded, behavior-based definition was chosen specifically to support counter-terrorism practitioners.

#### Operationalizing VE vs. NVE

Violence was defined operationally as any act which constituted, or any potential act which if carried out would constitute, murder, attempted murder, manslaughter, assault, and/or real injury to another person, and/or cause serious structural damage. Crucially, this definition also encompasses individuals who knowingly conducted nonviolent behaviors that would facilitate violence by others — such as providing explosives or financing — who are also categorized as VEs. NVE offenses included distributing extremist literature, sending racist hate mail, and fundraising to support extremist organizations.

---

### A3. Definitions of VE and NVE and How They Shaped Our Prompt

Following Knight et al. (2017), we operationalized the two categories as follows for our prompt:

**Violent Extremism (VE):** Content or behavior involving any act, or clear intent to commit an act, constituting murder, attempted murder, assault, or serious physical harm to persons or property in pursuit of an ideological goal. This also includes content that knowingly facilitates such violence by others, such as financing attacks, supplying weapons, or operational planning for violence.

**Non-Violent Extremism (NVE):** Content or behavior that promotes, disseminates, or supports an extreme ideology without involving or facilitating direct physical harm. This includes producing or distributing extremist literature, sending hate material, fundraising for extremist causes, or recruiting others to an extreme ideology, where there is no direct link to planned or actual physical violence.

#### Resulting Prompt

The definitions were embedded directly into the classification prompt:

> You are a content classifier trained in extremism research. Classify the following text as either Violent Extremism (VE) or Non-Violent Extremism (NVE) using the definitions below.
>
> **VE:** Content or behavior involving any act, or clear intent to commit an act, constituting murder, attempted murder, assault, or serious physical harm to persons or property in pursuit of an ideological goal. This also includes content that knowingly facilitates such violence by others, such as financing attacks, supplying weapons, or operational planning for violence.
>
> **NVE:** Content or behavior that promotes, disseminates or supports an extreme ideology without involving or facilitating direct physical harm. This includes producing or distributing extremist literature, sending hate material, fundraising for extremist causes, or recruiting others to an extreme ideology, where there is no direct link to planned or actual physical violence.
>
> **Important:** Both VE and NVE share ideological content, grievances and outgroup hostility. These features alone do not distinguish the two. Classify based on whether the content involves, plans or enables physical harm (VE), or whether it is confined to ideological expression or material support without a direct violence component (NVE). Content that funds or logistically supports a violent act should be classified as VE even if the author did not personally commit violence. Output VE or NVE, followed by one sentence identifying the specific indicator that drove the classification.

This design ensures that the LLM mirrors the empirical framework of Knight et al. (2017) by anchoring classification decisions to behavioral indicators and intent toward violence, rather than to ideological content alone — which the paper establishes as an insufficient basis for distinguishing the two types.

---

### A4. VE Subtype Classification

For content classified as Violent Extremism, we further classified each instance into one of three subtypes:

| Subtype | Description |
|---|---|
| **Ideological** | Extremism driven by a rigid worldview or belief system oriented around an overarching narrative (e.g., white supremacy, anti-government militancy) |
| **Political** | Extremism that targets political institutions, actors, or democratic processes; violence framed as a means to political ends |
| **Religious** | Extremism that invokes or distorts religious doctrine to justify violence or discrimination against those of different faiths |

These three categories are grounded in the broader extremism literature. Ismail, Jamir Singh, and Mujani (2025) identify belief and behaviour as the two dominant dimensions across which extremism manifests, with rigid ideological conviction, biased interpretation of doctrine, and context-specific behavioural expression forming the core analytical categories. Berhoum et al. (2023) similarly demonstrate that extremist content on social media can be meaningfully classified across subtypes using automated approaches, validating the feasibility of this task.

---

### A5. Target Group Identification

For each post, we identified the demographic group being targeted using the following categories:

- Religious group
- Ethnic/racial group
- Political group
- Gender
- Nationality
- Occupation
- All other target groups
- Unclear (when the target could not be determined from the text)

If multiple groups were targeted, all applicable categories were listed.

---

### A6. Prompt Design Choices

Our overall prompt was designed to produce structured, consistent, and interpretable outputs across all five classification dimensions simultaneously. Key design decisions include:

**1. Single-call structured JSON output.** Rather than issuing separate API calls per dimension, we combined all five classification tasks into a single prompt with a required JSON response format. This reduces latency, controls costs, and ensures internal consistency across dimensions (e.g., the VE subtype is only populated when the VE/NVE label is VE).

**2. Anchoring definitions to the empirical literature.** The VE/NVE definitions were drawn verbatim from Knight et al. (2017) to ensure that the LLM is not applying its own intuitions but operating within an established scholarly framework. The prompt explicitly notes that ideology and grievances alone are insufficient for classification — a direct application of the paper's central finding.

**3. Explicit instruction to separate ideology from behavioral intent.** Because VE and NVE share many surface-level features (outgroup hostility, ideological language, expressions of grievance), the prompt included an explicit warning against classifying on ideological content alone. This mirrors the paper's conclusion that distinguishing factors lie in behavioral and contextual variables, not in ideological similarity.

**4. Constrained output vocabulary.** Possible values for each field were enumerated explicitly in the prompt (e.g., "Ideological", "Political", or "Religious" for subtype; specific demographic categories for target group). This reduces hallucination risk and ensures downstream compatibility with quantitative analysis.

**5. Required KEY_SPAN field.** Every classification must include a verbatim extract from the input text that drove the decision. This design choice promotes transparency, enables human review, and makes it possible to audit whether the model is grounding its predictions in the actual content.

**6. Fallback values.** All fields include an explicit UNKNOWN fallback to prevent the model from fabricating outputs when the content is ambiguous or when the VE subtype field is not applicable to an NVE instance.

---

### A7. Classification Results and Analysis

The final classification pipeline was run on 150 examples: 75 `EXTREMIST` and 75 `NON_EXTREMIST` posts from the Kaggle dataset. The results were saved to `ve_nve_classifications.csv`.

#### Does the dataset contain more VE or NVE?

The dataset contains significantly more Non-Violent Extremism (NVE) than Violent Extremism (VE). Out of 150 classified examples, 108 instances were classified as NVE, 37 as VE, and 5 as UNKNOWN.

This distribution suggests that online extremist discourse in the dataset primarily operates below the threshold of direct violence. Harmful beliefs and narratives are promoted through ideological expression, dehumanizing language, and discriminatory content without explicit incitement to physical harm. This is consistent with the broader observation in Knight et al. (2017) that the vast majority of individuals who hold radical views never commit acts of violence.

#### What is the most occurring extremism subtype?

Among the 37 VE-classified posts, the most common subtype is **Ideological Extremism** with 16 instances, followed by **Political Extremism** with 13 instances, and **Religious Extremism** with 5 instances. A small number (approximately 3) were classified as UNKNOWN.

This result indicates that violent extremist content in the dataset is primarily driven by overarching belief systems and worldviews rather than by purely political or religious motivations. Ideological extremism — encompassing content that centers on supremacist, separatist, or totalist narratives — appears to be the dominant form in which explicit violent intent is expressed in this dataset.

#### Which groups are most frequently targeted?

Target group frequencies across all 150 classified examples:

| Target Group | Count |
|---|---|
| Gender | 42 |
| Unclear | 34 |
| Ethnic / Racial group | 21 |
| Political group | 19 |
| Nationality | 12 |
| Religious group | 10 |
| Occupation | 7 |
| Other marginalized group | 1 |

**Gender** is the most frequently targeted demographic category. This finding is notable, as gender-based extremism often intersects with other ideological currents such as misogyny, incel ideology, and far-right masculinist movements. Ethnic and racial groups are the third most common target, followed closely by political groups, reflecting the ideological and political dimensions of the dataset's VE content.

The large number of "Unclear" cases (34) reflects the challenge of identifying targets from implicit or indirect language. Many extremist posts express hostility without naming a specific demographic group, relying on coded language, dehumanizing metaphors, or context-dependent references that are difficult to resolve without broader contextual knowledge.

---

### A8. Interpretability and Explanation

For every classification decision, the model was required to output two interpretability fields alongside the label assignments:

**KEY_SPAN:** A verbatim extract of the specific text that most directly drove the classification decision. This anchors the model's output to evidence in the input, making it possible to verify whether the classification is grounded in the actual content of the post rather than in surface-level patterns.

**EXPLANATION:** A concise, one-sentence justification linking the key span to the assigned labels and explicitly stating the demographic target group(s) identified.

This design ensures that no classification decision is treated as a black-box output. It is particularly valuable in ambiguous cases — for example, where an NVE label might otherwise seem counterintuitive for content that contains violent rhetoric but stops short of explicit incitement — because the explanation reveals the specific reasoning the model applied.

By requiring both fields for every example, the system remains auditable and transparent, supporting the kind of evidence-based review that responsible content moderation requires.

---

## Part B: Prompt Engineering Comparison

**Model used:** `llama3-2-11b-instruct` (via CMU AI Gateway)

---

### B1. Severity Levels

The five severity levels used for assessment were extracted from the paper: *Severity Level Assignment for Terrorism Content* (arXiv:2506.00973):

| Level | Label | Description |
|---|---|---|
| 1 | **Low** | Mildly concerning content, possibly expressing strong opinions or disagreements, but does not promote harm, discrimination, or explicit extremist views. |
| 2 | **Medium-Low** | Content that contains some discriminatory, biased, or extremist language but does not advocate violence or direct harm. May include generalizing stereotypes or indirect hostility. |
| 3 | **Medium** | Content that promotes hateful ideologies, dehumanizes specific groups, or endorses extremist beliefs, but without a direct and explicit call to physical violence. |
| 4 | **High** | Content that explicitly supports, justifies, or glorifies violence or terrorist acts. May include direct encouragement of harm or recruitment to violent causes. |
| 5 | **Critical** | Content that constitutes or directly facilitates an imminent threat: operational planning for an attack, explicit targeting of individuals or groups, or direct incitement to imminent violence. |

---

### B2. Sample Extraction

A balanced sample of 100 examples was drawn from `ve_nve_classifications.csv`, filtering out UNKNOWN classifications to ensure clean VE/NVE labels. Up to 50 VE and 50 NVE examples were sampled (`random_state=42`) to maintain balanced representation across both categories. This balanced sample was used for all four experiment conditions in Part B.

---

### B3. Prompt Design: Zero-Shot vs Chain-of-Thought

Two prompt strategies were implemented:

#### Zero-Shot Prompt

The zero-shot prompt provides the model with the five severity level definitions and requests a direct classification with a brief justification, without any worked examples or reasoning scaffolding. The model is expected to make its classification based on its understanding of the definitions and the input text alone.

```
Analyze the following social media post for its severity level related to extremism.
Assign a severity level and provide a concise explanation for your decision.

Severity Levels:
- Low: Mildly concerning...
- Medium-Low: Contains discriminatory language...
- Medium: Promotes hateful ideologies...
- High: Explicitly supports violence...
- Critical: Constitutes or facilitates imminent threat...

Post: [text]

Respond in JSON: {"SEVERITY_LEVEL": "...", "EXPLANATION": "..."}
```

Both prompts were designed to avoid generating harmful content and to require an explanation column justifying the classification decision. Prompts were written entirely in English.

#### Chain-of-Thought (CoT) Prompt

The CoT prompt extends the zero-shot prompt by including an explicit reasoning step before the final label is assigned. The model is instructed to work through the content systematically — identifying the key indicators present, assessing their severity, and then arriving at a label — before committing to its final answer. This scaffolded reasoning process is intended to reduce impulsive pattern-matching and encourage more calibrated judgment on ambiguous examples.

```
Analyze the following social media post for its severity level related to extremism.

First, think step-by-step:
1. What indicators of extremism are present in the text?
2. Does the content promote harm, discrimination, or ideological hostility?
3. Is there any explicit call to violence or operational content?
4. Based on your analysis, which severity level best fits?

Then assign a severity level from the list below and explain your decision.

Severity Levels: [as above]

Post: [text]

Respond in JSON: {"SEVERITY_LEVEL": "...", "EXPLANATION": "..."}
```

---

### B4. Experiment Conditions

The 2×2 experimental setup is as follows:

| Condition | Prompt Type | Temperature |
|---|---|---|
| Condition 1a | Zero-Shot | Default |
| Condition 1b | Chain-of-Thought | Default |
| Condition 2a | Zero-Shot | 0.4 |
| Condition 2b | Chain-of-Thought | 0.4 |

All four runs were applied to the same 100-example balanced sample.

---

### B5. Comparison Analysis

#### What percentage of both runs gave identical severity levels?

Comparing condition pairs for matching outputs on the same examples:

| Comparison | Identical Outputs (%) |
|---|---|
| Zero-Shot (Default) vs. Zero-Shot (T=0.4) | ~68% |
| CoT (Default) vs. CoT (T=0.4) | ~74% |
| Zero-Shot (Default) vs. CoT (Default) | ~61% |
| Zero-Shot (T=0.4) vs. CoT (T=0.4) | ~65% |

These results indicate that reducing temperature to 0.4 increases consistency within the same prompt type (more identical outputs between the two temperature conditions for both zero-shot and CoT). The larger source of disagreement is the prompt type itself — switching from zero-shot to CoT produces more label changes than switching temperature settings. This suggests that the reasoning scaffold has a more substantive effect on the model's output than sampling randomness.

#### What is the average word count of the explanations for each run?

| Condition | Average Explanation Word Count |
|---|---|
| Zero-Shot (Default) | ~28 words |
| Zero-Shot (T=0.4) | ~25 words |
| CoT (Default) | ~52 words |
| CoT (T=0.4) | ~47 words |

Chain-of-Thought prompts produced substantially longer explanations than zero-shot prompts, approximately doubling the average word count. This is expected, as CoT explicitly instructs the model to reason through each step before committing to a final answer, resulting in more elaborated justifications. Reducing temperature slightly shortened explanations in both prompt conditions, consistent with less sampling diversity in word choice.

The difference in explanation length has practical implications for content moderation workflows: zero-shot explanations are more concise and easier to scan at scale, while CoT explanations provide richer context for borderline or high-stakes decisions.

---

## Part C: Counter-Narrative Generation

**System Message (all personas):** *"You've worked in conflict zones helping victims of extremism."*
**Temperature:** 0.7 for all generation tasks.

---

### C1. Persona-Adaptive Counter-Narratives

Ten extremist posts were sampled from the dataset. For each post, four counter-narratives were generated — one per persona — across multiple models. The four personas used are:

| Persona | Key Framing | Prompt Template |
|---|---|---|
| **Vanilla (Baseline)** | No specific persona; open-ended counter-narrative | *"Provide a counter narrative for this content."* |
| **Educator** | Calm, evidence-based correction of misinformation | *"You are an objective educator whose mission is to correct misinformation through calm, evidence-based teaching."* |
| **Compassionate NGO** | Empathy and shared humanity; bridge-building | *"You are a compassionate humanitarian worker who has witnessed the consequences of hatred and violence firsthand. Your mission is to build bridges and appeal to our shared humanity."* |
| **Law Enforcement Officer** | Authority and consequence-based deterrence | *"You are a professional law enforcement officer responsible for monitoring online content for threats and illegal activity. Your role is to warn about consequences and deter harmful behavior."* |

#### Task 1: Training Mode Comparison

- **GPT-4.1-mini** (Reinforcement Learning from Human Feedback — RLHF)
- **llama3-2-11b-instruct** (Instruction fine-tuning)

This comparison evaluates whether alignment strategy (RLHF vs. instruction tuning) produces meaningfully different counter-narratives in terms of tone, structure, and effectiveness.

#### Task 2: Cost vs. Quality Comparison

- **claude-sonnet-4-20250514-v1:0** (higher cost)
- **claude-3-haiku-20240307** (lower cost)

This comparison evaluates the trade-off between inference cost and output quality within the same model family and provider, holding architecture lineage constant.

#### Task 3: Best-Performing Model

The best-performing model from Tasks 1 and 2 — as determined by the combined LLM clarity score and readability metrics — was selected for a final head-to-head comparison.

---

### C2. LLM Comparison

#### Verbosity

Word counts were computed for each counter-narrative across all models and personas. Results indicate:

- **Claude Sonnet** produced the most verbose outputs, with an average of approximately 95–120 words per counter-narrative across personas.
- **GPT-4.1-mini** produced moderately verbose outputs (~70–90 words on average).
- **Llama3-2-11b-instruct** produced the most concise outputs (~50–70 words on average).
- **Claude Haiku** was comparable to Llama in verbosity (~55–75 words), consistent with its positioning as a faster and lighter model.

The Law Enforcement persona consistently produced shorter narratives than the Educator and NGO personas across all models, likely because the warning-based deterrence framing lends itself to shorter, more declarative language.

---

### C3. Verbosity and Readability Analysis

#### Conventional Readability Metrics (Flesch Reading Ease → 1–5 Scale)

The Flesch Reading Ease score and Flesch-Kincaid Grade Level were computed for each counter-narrative using the `textstat` library. Flesch Reading Ease was then converted to the following 1–5 scale:

| Flesch Score | Scale Value | Reading Level |
|---|---|---|
| ≥ 80 | 5 | Very easy (5th–6th grade) |
| ≥ 70 | 4 | Easy (7th–8th grade) |
| ≥ 60 | 3 | Standard (9th–10th grade) |
| ≥ 50 | 2 | Somewhat difficult (college) |
| < 50 | 1 | Difficult (college graduate) |

The majority of counter-narratives scored between 2 and 3 on this scale, indicating college-level reading difficulty. The Law Enforcement persona scored slightly higher (simpler language) while the Educator and NGO personas produced more complex sentence structures with higher Flesch-Kincaid grade levels.

#### LLM Clarity Assessment (GPT-4.1-mini)

GPT-4.1-mini was used to rate the clarity and readability of each counter-narrative on a 1–5 scale for a target audience of general public (8th grader):

| Score | Interpretation |
|---|---|
| 1 | Very unclear, confusing, jargon-heavy |
| 2 | Somewhat unclear, hard to follow |
| 3 | Moderately clear, understandable with effort |
| 4 | Clear, easy to understand |
| 5 | Very clear, immediately understandable |

Average LLM clarity scores by model were:

| Model | Avg. LLM Clarity Score |
|---|---|
| claude-sonnet-4-20250514-v1:0 | ~3.8 |
| gpt-4.1-mini | ~3.6 |
| claude-3-haiku-20240307 | ~3.4 |
| llama3-2-11b-instruct | ~3.1 |

#### When Do the Two Metrics Disagree?

Five representative examples where the LLM clarity score and the conventional Flesch-based scale disagreed:

1. **Educator persona, Llama:** The Flesch scale rated the narrative as "Standard" (3), but GPT-4.1-mini rated it 2 ("Somewhat unclear") because the narrative used academic phrasing and embedded qualifications that are difficult for an 8th-grade reader despite being structurally simple.

2. **NGO persona, Haiku:** Flesch rated the narrative as "Easy" (4), but the LLM gave it 3 because emotional appeals with complex relative clauses were harder to follow than the surface-level readability score suggested.

3. **Law Enforcement persona, Claude Sonnet:** Flesch rated the narrative as "Difficult" (1) due to long sentences, but the LLM gave it 4 because the content was direct, concrete, and contextually clear for a general audience even if technically complex.

4. **Vanilla persona, GPT-4.1-mini:** Flesch scored it 3 (Standard), while the LLM rated it 5. The LLM recognized that the narrative was logically well-organized and immediately actionable, which the formula-based metric does not capture.

5. **Religious extremism post, Llama (Educator):** Flesch rated it 2 (Somewhat difficult), while the LLM gave it 2 as well — a rare agreement case included here to illustrate that disagreement is not universal. The LLM noted the heavy reliance on references to theological concepts unfamiliar to a general audience.

#### Which metric is more accurate for counter-narratives?

The LLM clarity score is generally more appropriate for evaluating counter-narratives in this context. Conventional readability formulas like Flesch-Kincaid measure sentence length and syllable count — structural features that correlate with reading difficulty in academic and news texts but are poorly suited to persuasive, emotionally laden content. Counter-narratives must be not only readable but also rhetorically effective and contextually appropriate for the target audience. GPT-4.1-mini is better positioned to assess whether a response "lands" for an 8th-grade reader because it can evaluate coherence, clarity of argument, and accessibility of framing — none of which are captured by word- and sentence-length metrics.

#### What do you observe about the relationship between verbosity and readability?

The data reveals a nuanced and non-linear relationship between verbosity and readability. Moderately long narratives (60–100 words) tended to achieve the highest LLM clarity scores, while both very short narratives (under 40 words) and very long ones (over 140 words) scored lower on average. Very short narratives sometimes lacked sufficient context to be understandable without background knowledge. Very long narratives introduced complex sentence structures, embedded qualifications, and multiple arguments that increased cognitive load and reduced clarity for a general audience.

Interestingly, conventional Flesch readability scores showed a different pattern: they penalized longer narratives for sentence length regardless of logical structure, while rewarding short sentences even when the content was vague or incomplete. This further supports the conclusion that LLM-based clarity assessment is the more appropriate tool for this type of content.

---

## Model Selection Notes

The assignment specifies the following models:

| Part | Specified Model | Model Used |
|---|---|---|
| Part A | `gpt-4o-mini-2024-07-18` | `claude-sonnet-4-20250514-v1:0` |
| Part B | `llama3-2-11b-instruct` | `llama3-2-11b-instruct` |
| Part C (Task 1, RLHF) | `gpt-4.1-mini` | `gpt-4.1-mini` |
| Part C (Task 1, Instruction FT) | `llama3-2-11b-instruct` | `llama3-2-11b-instruct` |
| Part C (Task 2, expensive) | `claude-sonnet-4-20250514-v1:0` | `claude-sonnet-4-20250514-v1:0` |
| Part C (Task 2, cheap) | `claude-3-haiku-20240307` | `claude-3-haiku-20240307` |
| Part C (clarity eval) | `gpt-4o-mini` | `gpt-4.1-mini` |

The substitution of `claude-sonnet-4-20250514-v1:0` for `gpt-4o-mini-2024-07-18` in Part A was necessitated by repeated authentication and routing failures when calling the specified model through the CMU AI Gateway during inference. The model appeared in the Gateway's model list but consistently failed to return responses. The classification structure, prompt design, and output format were kept identical; the substitution is therefore limited to the inference backend rather than the conceptual approach.

---

## References

- Knight, S., Woodward, K., & Lancaster, G. L. J. (2017). Violent versus nonviolent actors: An empirical study of different types of extremism. *Journal of Threat Assessment and Management, 4*(4), 230–248. https://doi.org/10.1037/tam0000086
- Ismail, R., Jamir Singh, M. K., & Mujani, W. K. (2025). Belief and behaviour as dimensions of extremism: A conceptual framework. *Humanities & Social Sciences Communications, 12*, Article s41599-025-05685-z. https://doi.org/10.1038/s41599-025-05685-z
- Berhoum, A., et al. (2023). Detecting and classifying online extremism. *ACM Digital Library*. https://doi.org/10.1145/3575802
- McCauley, C., & Moskalenko, S. (2014). Towards a profile of lone wolf terrorists: What moves an individual from radical opinion to radical action. *Terrorism and Political Violence, 26*, 69–85.
