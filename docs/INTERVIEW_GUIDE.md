# Interview guide: 10-K Risk Radar

This is the "teach me so I can defend it" companion to the README. Read it top to bottom
once; then use the question bank at the end to rehearse. Every number quoted here is in
`data/results/` and can be regenerated with the commands in the README.

---

## 1. The 60-second version

> "I built a pipeline that pulls 176 annual reports (Form 10-K) for 44 US insurers, insurance
> brokers and banks from SEC EDGAR, finds the Risk Factors section in each, splits it into
> about 5,800 individual risk factors, and classifies each one into a 14-category risk
> taxonomy with a local LLM. The centerpiece is the evaluation: I wrote labeling
> guidelines, hand-labeled 240 risk factors, and compared four approaches - a keyword
> baseline, embeddings + logistic regression, and zero-/few-shot prompting at two model
> sizes. The 14-billion-parameter model zero-shot won with 0.91 macro-F1 on a held-out
> test set of 180, versus 0.75 for keywords. Then I used the classified data and an
> embedding-based year-over-year diff to show how fast AI-risk disclosures spread - and I
> cross-checked that trend against SEC's own full-text search."

If they want one sentence: *"An LLM extraction pipeline over SEC filings where the
interesting part is how I proved it works."*

---

## 2. Domain primer: what 10-K risk factors are (risk-analyst angle)

- **Form 10-K** is the annual report every US public company files with the SEC.
  **Item 1A, "Risk Factors"** is where the company lists the material risks to its
  business. The rule behind it is **Regulation S-K Item 105**.
- Item 105 requires each risk factor to have a **descriptive sub-caption** (heading),
  to be organized under relevant headings, and, if the section is longer than 15 pages,
  a **risk factor summary** of no more than two pages at the front. Generic risks that
  apply to any company are supposed to go at the end under "General Risk Factors".
  My segmenter relies on the first rule (headings) and deliberately skips the summary
  (it duplicates the real risk factors).
- **Why companies write them:** legal protection. Disclosing a risk is a defence if it
  later happens (the "safe harbor" / "we told you" logic), which is why the language is
  conservative and boilerplate-heavy - and why *changes* year over year are more
  informative than the text itself.
- **Who reads them:** equity and credit analysts, rating agencies, regulators, and
  underwriters (a D&O underwriter reading a company's cyber and litigation disclosures),
  and internal risk teams benchmarking their own disclosures against peers.
- **Carrier vs broker vs bank:** carriers (P&C and life insurers) take risk onto their
  balance sheet, so their risk factors are dominated by catastrophe, reserving, investment
  and reinsurance risk. **Brokers** (Marsh McLennan, Aon, Gallagher, Brown & Brown, WTW)
  don't carry insurance risk; their risks are E&O liability, talent, M&A integration,
  regulation of how intermediaries are paid, and technology. **Banks** emphasise credit,
  interest-rate, liquidity and regulatory capital risk. You can see exactly this in
  `docs/category_mix_by_sector.png`.
- **What I would NOT conclude from this data:** that a company is *riskier* because it
  lists more risks. Risk factors measure *disclosure*, shaped by lawyers, SEC comment
  letters and peer imitation - not exposure. More AI risk factors means companies feel
  obliged to talk about AI, not that AI losses happened.

---

## 3. Architecture and every design decision

```
EDGAR (submissions API) -> cached 10-K HTML -> blocks -> Item 1A section -> risk factors
   -> {keyword | embeddings+LR | LLM zero/few-shot} labels -> YoY diff -> insights + app
                                     ^
                                gold set (240 hand labels) -> metrics, CIs, error analysis
```

### 3.1 Fetching (src/riskradar/edgar.py)
- **Submissions API** (`data.sec.gov/submissions/CIK##########.json`) lists every filing
  with form type, accession number, report date and primary document. I filter
  `form == "10-K"` (excludes 10-K/A amendments) and map report date to fiscal year.
- **Gotcha I hit:** large banks (JPMorgan, BofA, Citi, Wells, Goldman, Morgan Stanley) file
  thousands of prospectus supplements (424B2), so their older 10-Ks are not in the
  "recent" block. The code walks the paginated side files. Without this I silently had
  only 2025 filings for six banks - a good example of why I check counts per company.
- **SEC fair access:** max 10 requests/second and a User-Agent with contact info,
  otherwise 403. My client refuses to be configured above 10 req/s or without an email,
  defaults to 5 req/s, retries 429/5xx with exponential backoff, and caches every
  response gzipped on disk so re-runs never re-download. When extraction runs in 4
  processes, each gets 1/4 of the rate budget so the *total* stays under the cap.
- Ticker gotcha: Marsh McLennan's ticker is now MRSH (not MMC) in SEC's ticker file.

### 3.2 Finding Item 1A (src/riskradar/extract.py) - extraction success 176/176
- HTML is flattened into **blocks** (paragraph-level text) that keep formatting:
  fraction bold/italic/underlined, the emphasized lead-in text, and a typographic
  signature (font size, colour, family). Inline XBRL hidden facts (`ix:header`) and
  `display:none` are dropped.
- **Strategy cascade, recorded per filing:**
  1. `item1a` (160 filings): find "Item 1A ... Risk Factors" headings and the next
     "Item 1B/1C/2" heading. The table of contents contains the same headings, so I score
     every candidate span and keep the **longest** - the TOC span is tiny.
  2. `heading` (8: Citigroup, Morgan Stanley): their 10-Ks are laid out like annual
     reports; I look for a standalone "RISK FACTORS" caption and end at the next known
     top-level section. Bug I fixed: "Risk Management Strategies, Models and Processes"
     is a sub-heading *inside* Morgan Stanley's risk factors and was ending the section
     early, so section captions must match whole, not as prefixes.
  3. `exhibit13` (8: Wells Fargo, U.S. Bancorp): Item 1A just says "see the Annual Report",
     which is filed as **Exhibit 13**. I read the filing index page, find the EX-13
     document, and apply strategy 2 there.
- **Validation:** automatic (176/176 located, min section 26k characters) plus a manual
  spot check of 25 random sections' first and last blocks: 25/25 boundaries correct
  (`data/results/extraction_spotcheck.csv`). 
- Honest caveat: "located" is not "perfect". A section can include a stray page footer;
  segmentation removes repeated headers/footers and page numbers.

### 3.3 Segmenting into risk factors (src/riskradar/segment.py) - 5,823 risk factors
- Three heading styles exist in the wild: standalone bold/italic paragraphs (5,117),
  **run-in lead-ins** where the bold sentence starts the paragraph (Travelers; 592), and
  **colour/font-only** headings with no bold at all (The Hartford; 114, found by a
  typographic-outlier fallback that only runs when emphasis finds < 8 risk factors).
- Short captions without a verb or period ("Insurance-Related Risks") are **group
  headers** - kept as context, not as risk factors. A verb list distinguishes MetLife's
  Title-Case headings ("We May Lose Business Due to a Downgrade...") from group headers.
- Page-break repair: a block starting lower-case after a block without final punctuation
  is a continuation and is merged back.
- `<br><br>` inside one `<div>` is a paragraph break (KeyCorp 2025 put each heading after
  two line breaks in the previous paragraph's div; fixing it took KeyCorp from 13 to 38
  risk factors).
- Gold-set check: all 240 sampled units were judged real single risk factors
  (`segment_ok = 1`), judged from heading + first ~1,200 characters - so a factor glued to
  the next one later in its body could slip through.

### 3.4 Taxonomy (src/riskradar/taxonomy.py)
14 categories built for insurers + banks: cyber_tech, ai_model, climate_catastrophe,
regulatory_legal, market_interest_rate, credit, reserving_underwriting, liquidity_capital,
reinsurance, pandemic_health, macro_geopolitical, competition_strategy,
operational_people, other. Each has a definition, includes, excludes and keywords; the
same object feeds the prompt, the keyword baseline, the guidelines and the app - a single
source of truth so they can't drift.

Why **single primary + optional secondary** instead of multi-label? Most risk factors
touch several topics ("a cyberattack could cause regulatory fines and reputational harm").
Multi-label gold is far slower and noisier to annotate. Primary = *the loss driver named
in the heading*; secondary captures the rest. "Lenient accuracy" (prediction matches the
primary or a secondary) quantifies how many errors are boundary disagreements.

### 3.5 Classifiers
| approach | how | why it's in the comparison |
|---|---|---|
| keyword | count category keywords, heading hits x3, argmax | the floor; free, transparent, auditable |
| embeddings + LR | nomic-embed-text vectors, class-balanced logistic regression, out-of-fold predictions from 5-fold CV over all 240 gold items | the classic "small supervised model" answer |
| zero-shot LLM | taxonomy definitions + decision rules + JSON schema in the system prompt | no training data needed |
| few-shot LLM | same + 14 labeled examples (one per class) drawn from risk factors **not** in the gold set | tests whether examples help |

- **Structured output:** the endpoint is asked for JSON (`response_format=json_object`);
  output is parsed, keys are coerced ("Cyber Tech" -> cyber_tech), and anything off-taxonomy
  is counted as **invalid** and mapped to "other" - so failures lower the score instead
  of being dropped. Invalid rate is reported (0.6% for the winning config).
- **Constrained decoding vs free-text parsing** (talking point from the plan): JSON mode
  guarantees *syntactically* valid JSON; a JSON-schema / grammar (llama.cpp GBNF, Ollama
  `format` schema) can also force the category to be one of the 14 enum values. I kept JSON
  mode + validation because it works on any OpenAI-compatible endpoint; schema-constrained
  decoding is the next step and would push the invalid rate to zero.
- **Temperature 0 + response cache** (sqlite keyed by model+messages) makes every eval
  reproducible and re-runs free.
- **Provider-agnostic:** one small client targets any OpenAI-compatible base URL, so the
  same code runs against local Ollama or a hosted API by changing three env vars.

### 3.6 Year-over-year change detection (src/riskradar/diff.py)
- Embed every risk factor (heading + first 1,500 chars). For each company and pair of
  consecutive years, solve a **one-to-one assignment** (Hungarian algorithm) maximizing
  total cosine similarity; pairs below 0.80 are not matches.
- Unmatched this year = **new**; unmatched last year = **removed**; matched pairs are
  compared with a word-level diff: >= 0.97 similarity ratio = **unchanged**, >= 75 net
  words added and >= 25% longer = **expanded**, else **modified**.
- Why embeddings and not exact string match? Companies re-word headings every year;
  exact matching would call most risk factors "new". Why one-to-one? Without it, two new
  AI risk factors could both "match" last year's single technology factor.

---

## 4. Evaluation, the centerpiece

### 4.1 Gold set design
- 240 risk factors, **stratified** 60 per sector, random within sector (seeded), shuffled.
  60 dev (prompt iteration) / 180 test (final numbers only).
- Labels follow written guidelines (`docs/LABELING_GUIDELINES.md`): primary = loss driver
  in the heading, 14 ordered decision rules for known boundary cases, worked examples.
- **Who labeled:** the labels were produced by an AI assistant (Claude) acting as the
  annotator under those guidelines, not yet audited by me. Be upfront about this: it is a
  limitation (see section 6) and the first thing to fix is to re-label a sample yourself.
- **Consistency:** a blind second pass on a random 50-item subset agreed 49/50 (Cohen's
  kappa 0.98). Same annotator, near-duplicate items across years, so this is an
  optimistic upper bound on label reliability - not inter-annotator agreement.

### 4.2 Metrics, in plain English
- **Accuracy / micro-F1:** fraction of items labeled correctly (identical for single-label).
- **Macro-F1:** compute F1 per class, then average with equal weight. A model that nails
  the big classes (regulatory, competition) but misses rare ones (AI, reinsurance) gets
  punished. That's what we care about: the rare classes are the interesting ones.
- **F1 = 2PR/(P+R)**: harmonic mean of precision (of the items I called X, how many were X)
  and recall (of the true Xs, how many I found).
- **Cohen's kappa:** agreement corrected for chance agreement.
- **Bootstrap 95% CI:** resample the 180 test items with replacement 1,000 times,
  recompute macro-F1, take the 2.5th/97.5th percentiles. With 180 items the CI is about
  +/- 4-5 points, so differences smaller than that are not real.
- **Paired bootstrap:** resample the *same* items for two models and look at the
  difference - more powerful than comparing two separate CIs.

### 4.3 Results (test split, n = 180)
| approach | accuracy | macro-F1 [95% CI] | mean latency / item |
|---|---|---|---|
| keyword | 0.72 | 0.75 [0.67-0.80] | ~0 |
| embeddings + LR (5-fold CV) | 0.80 | 0.75 [0.69-0.85] | embedding only |
| qwen2.5 7B zero-shot | 0.80 | 0.81 [0.73-0.86] | 1.0 s |
| qwen2.5 7B few-shot | 0.82 | 0.84 [0.76-0.88] | 1.7 s |
| **qwen2.5 14B zero-shot** | **0.90** | **0.91 [0.86-0.94]** | 2.1 s |
| qwen2.5 14B few-shot | 0.87 | 0.89 [0.83-0.93] | 4.1 s |

How to talk about it:
1. **Model size matters more than prompting here.** 14B beats 7B by ~10 macro-F1 points
   (paired bootstrap: 14B zero-shot better in 99.9% of resamples).
2. **Few-shot helped the 7B but not the 14B** (14B zero vs few: +0.017, CI crosses zero ->
   no real difference) while doubling prompt tokens (1.9k -> 4.0k) and latency. So
   zero-shot 14B is the cost-efficient choice - and it's what classifies the full corpus.
3. **The keyword baseline is strong** (0.75). Disclosure language is formulaic, so keywords
   get a long way; the LLM's gain is on boundary cases (accounting estimates vs market
   risk, capital vs regulation) where rules in the prompt matter.
4. **Embeddings + LR has the same macro-F1 as keywords but better accuracy**: it learns the
   big classes but has only 2-6 training examples for the rare ones.

### 4.4 Error analysis (18 errors of the 14B zero-shot model)
`data/results/error_analysis_test.csv`:
- **10 of 18** errors predicted the gold *secondary* label - i.e. defensible boundary
  calls, not nonsense (lenient accuracy 95.6%).
- **3** were on items I had already flagged ambiguous while labeling (e.g. leased-asset
  residual value: market or credit risk?).
- **5** plain misses, e.g. a surety-bond loss risk called credit instead of underwriting,
  and derivative collateral calls called credit instead of liquidity.
- Most-confused areas: *accounting estimates / valuation judgments* ("other") vs
  market/credit risk; *competition_strategy* vs "other". These are partly taxonomy
  problems - a dedicated "accounting & reporting" class would remove most of them.

### 4.5 Prompt iteration log (dev split only)
v2 (taxonomy definitions only) -> v3 (+ decision rules distilled from the labeling
guidelines, written after looking at dev errors). 14B few-shot dev macro-F1 0.837 -> 0.856,
14B zero-shot 0.786 -> 0.841. I then froze v3 and ran the test split once. Files:
`data/results/prompt_iterations/`. Interview point: *the test set was never used to tune.*

### 4.6 Cost and latency
- Local GPU (RTX 4080, Ollama) so the API bill is $0; the real costs are GPU time and
  tokens. Zero-shot 14B: ~1.9k prompt + ~60 completion tokens per risk factor, ~33 risk
  factors per filing -> roughly 65k tokens per filing. See `data/results/classify_run.json`
  for measured corpus throughput.
- To price it on a hosted API: tokens per filing x the provider's per-token price. Prompt
  caching would cut cost a lot because the 1.9k-token system prompt is identical for
  every call.

### 4.7 Severity signal: what I learned
The LLM also outputs a 1-5 severity. With no gold labels for severity, I measured
agreement between the 7B and 14B models instead: quadratic-weighted kappa 0.18 zero-shot,
0.57 few-shot. That's low, so **I don't use severity in any conclusion**. Lesson: an LLM
will always return a number; whether the number means anything needs its own evaluation.

---

## 5. Findings and how to present them

(Exact values in `data/results/insights_summary.json`, `fts_crosscheck.csv`.)
- **AI-risk disclosure spread fast.** Per SEC full-text search, the share of the 44
  companies whose 10-K mentions "artificial intelligence" went from 19/44 (FY2022) to
  44/44 (FY2025); "generative AI" from 0 to 24. Inside the Item 1A text I extracted, the
  counts are 15 -> 34 and 0 -> 22, and every Item 1A mention is also an EFTS hit (no
  contradictions) - an independent check that extraction isn't inventing the trend.
- See the README for the LLM-classified view (filings with at least one AI-related risk
  factor by sector), cyber share of risk-factor text, and new risk factors by category.
- **Framing:** "This measures what companies *say*, not what happens to them."

---

## 6. Limitations - say these before they ask

1. **Gold labels were produced by an AI annotator (Claude) under written guidelines,
   not yet audited by a human**, and the consistency check is same-annotator. The
   guidelines and every label are committed so anyone can audit them.
2. 180 test items: rare classes have 1-5 examples, so per-class F1 for AI, pandemic and
   reinsurance is very noisy. A targeted rare-class sample would help.
3. Near-duplicates: companies repeat risk factors across years, so a few gold items are
   near-identical texts from different years.
4. The taxonomy is mine; another reasonable taxonomy would move the numbers.
5. Universe is hand-picked (44 large US filers), not a random sample of insurers/banks.
6. YoY "new" depends on a similarity threshold (0.80) that I set by inspecting
   distributions, not by labeled pairs.
7. Severity is unvalidated (see 4.7).

## 7. What I'd do next
- Human re-label of 50 items for true inter-annotator agreement; expand rare classes.
- JSON-schema constrained decoding (enum categories) and a hosted-model comparison.
- Label 50 matched/unmatched pairs to tune the YoY threshold with precision/recall.
- Add an "accounting & reporting" class; multi-label evaluation.
- Schedule quarterly refresh as new 10-Ks land; alert when a peer adds a new risk factor.

---

## 8. Question bank (with strong answers)

**Q: Why use an LLM at all if keywords get 0.75?**
A: Because the errors are concentrated where it matters. Keywords can't tell "regulators
raising capital requirements" from "we may be unable to raise capital"; they both say
*capital*. The 14B model gets 0.91 macro-F1 and its errors are mostly secondary-label
disagreements. But I'd keep the keyword model as a cheap monitor - if the two start
disagreeing a lot on new filings, something drifted.

**Q: How do you know your gold labels are right?**
A: I don't know they're perfect - I know how they were made. Written guidelines with
ordered decision rules, worked examples, notes on every ambiguous call, and a blind
re-label of 50 items (98% agreement, but same annotator, so optimistic). The honest next
step is a second human annotator and kappa between people.

**Q: Isn't 180 items too small?**
A: For the headline number it's enough to separate the models - the 14B vs keyword gap is
+0.17 macro-F1 with a paired-bootstrap CI of roughly [0.10, 0.23]. It is too small for
per-class claims on rare classes, and I say that.

**Q: Did you tune on the test set?**
A: No. Prompt changes were made on the 60-item dev split; the test split was scored once
after freezing prompt v3. The few-shot examples come from risk factors outside the gold set.

**Q: Why did few-shot not help the 14B?**
A: The bigger model already follows the definitions and rules; examples mostly add
tokens. Few-shot also anchors the model on the examples' surface features. It helped the
7B, which needed more guidance.

**Q: What does macro vs micro F1 tell you here?**
A: Micro is dominated by frequent classes (regulatory, competition, liquidity). Macro
treats AI and reinsurance equally with regulatory - and those rare classes are the
reason this project exists.

**Q: What's the hardest part of extracting Item 1A?**
A: Layout variety. The same legal section shows up as a standard Item 1A, as a heading in
an annual-report-style 10-K (Citi, Morgan Stanley), or only inside Exhibit 13 (Wells
Fargo, U.S. Bancorp). And the table of contents contains the same "Item 1A" text, so
naive regex grabs the TOC. I record which strategy worked for every filing.

**Q: How would you productionise this?**
A: Schedule the fetch for new 10-K filings, keep the cache, classify only new risk factors
(the LLM cache already makes reruns free), monitor invalid-output rate and keyword-vs-LLM
disagreement as drift signals, store results in a warehouse table, and keep a small
labeled sample refreshed each year to re-measure accuracy.

**Q: How is this relevant to AI governance?**
A: It's an example of governing an LLM inside a process: a documented taxonomy, written
labeling rules, a held-out test set, confidence intervals, error analysis, an invalid
output metric, and an explicit decision not to use an unvalidated output (severity). That
is model risk management (think SR 11-7 style validation) applied to an LLM - and the
data itself shows how fast financial firms started disclosing AI risk.

**Q: Why local models?**
A: Cost (zero marginal API spend for ~6k classifications plus evaluation reruns),
reproducibility (pinned model weights), and it shows the code is provider-agnostic.
The same client works with a hosted model by changing the base URL.

**Q: What would you NOT conclude from these charts?**
A: That banks or insurers became riskier, or that AI caused losses. Risk factors are legal
disclosures; they move with regulation, SEC comments and peer behaviour.
