# M4 data pipeline

Prepares the English subset of the [M4 benchmark](https://github.com/mbzuai-nlp/M4) (human vs. machine-generated text) for training and explaining the detectors.

```
load_m4  ->  preprocess  ->  split  ->  save_splits
(sec. 2)     (sec. 3)      (sec. 4)    (sec. 4)
```

Entry point: `prepare_data(cfg)` in `m4.py`. **All settings live in `configs/default.yaml`** (`data:` section plus the top-level `seed`); `m4.py` contains no hard-coded parameters (full reference in section 5).

```bash
uv sync 
just submodules 
just data  
uv run python main.py data --config configs/default.yaml
```

---

## 1. Raw data

Source: git submodule `external/M4`. 
Raw files: `external/M4/data/<source>_<generator>.jsonl` contain a list of JSON objects per line with one human text + one machine text per record.

The files are **not uniform**. Findings from inspecting all files:

| Issue | Where | Consequence |
|---|---|---|
| A blank line at the end of the file (3001 lines for 3000 records) | `*_bloomz.jsonl`, `*_dolly.jsonl` | `json.loads("\n")` raises `JSONDecodeError: Expecting value: line 2 column 1` |
| Human text stored under different keys: `human_text`, `text`, `abstract` | among files from different domains | one fixed key does not work |
| `machine_text` **starts with the prompt** (e.g. `Write an abstract for a Wikipedia article ... Abstract:\n...`) | the whole `arxiv/reddit/wikihow_bloomz`, ~half of `wikipedia_bloomz` | a detector would learn the prompt, not the style of the generator |
| `prompt` field is a template with `{title}` instead of the real prompt | `wikipedia_bloomz` | cannot strip the prompt by subtracting the `prompt` field |
| `human_text` / `machine_text` are **lists of reviews**, not strings | `peerread_chatgpt/cohere/davinci/dolly/llama` | not plain text |
| Completely different schema (`human_reviews`, `chatgpt_reviews`, ...), no `machine_text` | `peerread_bloomz` | unusable as-is |
| A few prompt echoes also in non-bloomz files | `wikihow_dolly2` (25 records) | need a text-level filter |
| Same generator has different `model` strings in different files | `cohere`, `cohere-xlarge-nightly`, `command-xlarge-nightly` | one family split into 3 "generators" |
| Non-English files | `arabic_*`, `bulgarian_*`, `russian_*`, `urdu_*`, `qazh_*`, `germanwikipedia_*`, `id-newspaper_*` | out of the project scope |

---

## 2. Loading — `load_m4(data_config)`

Responsibility: turn the raw files into one flat DataFrame, one row per labeled text with some metadata.

Output columns:

| Column | Meaning |
|---|---|
| `text` | the text (may still be a non-string / dirty at this stage) |
| `label` | `0` = human, `1` = machine |
| `source` | domain, from the file name prefix: `wikipedia`, `reddit`, `wikihow`, `arxiv` |
| `model` | raw generator name (`r["model"]`, falling back to the file-name suffix); `"human"` for label 0 |
| `family` | normalised generator family (`chatgpt`, `cohere`, `davinci`, `dolly`, `flan-t5`); `"human"` for label 0 |

Each raw record yields **two rows** (its human text and its machine text).

### Decisions

**2.1 Files are selected by file name, not by content.**
`<source>_<generator>.jsonl` is split with `partition("_")`. Simple, deterministic, and avoids opening files we do not need.
(`wikihow_dolly2` gives generator suffix `dolly2`, which is only a fallback: `model` normally comes from the record.)

**2.2 Sources: `data.domain` = `[wikipedia, reddit, wikihow, arxiv]`.**
These are the English sources whose files have plain-text human and machine fields.
`peerread` is **excluded** because its fields are lists of reviews (see table above); using it would need separate parsing and the text type (peer review) differs a lot from the others. It can be added later if more data is needed.

**2.3 Generators: `data.excluded_models` = `[bloomz]`.**
`bloomz` echoes the prompt in `machine_text`. Restricting to datasets with clean machine text was chosen over trying to strip the prompt, because
(a) for `wikipedia_bloomz` the `prompt` field is a template, so exact subtraction is impossible,
(b) regex stripping can cut real content, and a silent bug here would leak the prompt as a label shortcut.
<!-- **Way back if we run out of data:** remove `bloomz` from `excluded_models` and read the clean generated text from `machine_abstract` / `machine_answer` (present in the bloomz files, verified only on `wikipedia_bloomz`), instead of `machine_text`. -->

**2.4 Human text key: first non-empty of `human_text`, `text`, `abstract`** (`data.human_text_keys`). The machine key is `data.machine_text_key` (`machine_text`).
Order matters: `human_text` is the standard key, so it wins if a file ever has several.

**2.5 Blank lines are skipped** (`if not line.strip(): continue`). Fixes the `JSONDecodeError` above.

---

## 3. Preprocessing — `preprocess(df, data_config, seed)`

Input: the DataFrame from `load_m4`. Output: cleaned, de-duplicated, **class-balanced** DataFrame with the same columns.
Steps run in this order; the order is intentional.

| # | Step | Config |
|---|---|---|
| 1 | drop rows whose `text` is not a `str` | — |
| 2 | collapse whitespace, strip, drop empty texts | — |
| 2b | strip leading separator junk (`; ,`, `,`), drop emptied texts | `data.strip_leading_regex` |
| 3 | drop **machine** texts that start with an echoed prompt | `data.prompt_prefixes` |
| 3b | drop texts shorter than `min_words` words | `data.min_words` |
| 4 | drop texts that occur with **both** labels | — |
| 5 | drop exact duplicates (keep first) | — |
| 6 | `undersample`: balance classes, sources, generator families; cap total size | `data.max_samples`, `seed` |

### Decisions

**3.1 Non-string texts are dropped (step 1)** — covers records where the human or machine field is missing or is a list. Done before any `.str` operation, which would otherwise fail or produce NaN.

**3.2a Whitespace is collapsed before everything else (step 2a)**, so that deduplication and the word count are not fooled by `\n`, double spaces or trailing spaces. The collapsed text is what is stored: the detector sees single-spaced text. Newlines/paragraph structure are therefore **lost** (accepted: the detector tokenises text anyway; revisit if formatting turns out to be a useful signal or a leak).

**3.2b Leading separator junk is stripped (step 2b).** Some texts started with `; ,` (and 61 more with `,`), **all of them human `wikihow` texts**. Because only human texts have it, it would cause a label leak in the next project part with detectors training or experiments.

**3.3a Prompt-echo filter on machine texts only (step 3a).**
Regex built based on config file, applied only to `label == 1`.
Reason for the label restriction: human texts can legitimately start with "Write..." (e.g. WikiHow articles); a filter on all rows would remove real human data and bias the classes. The filter is a safety net for the files that are not excluded (`wikihow_dolly2`); the main protection is excluding `bloomz`.
**Known limitation**: only these three prefixes are covered. After changing sources/generators, re-check `machine_text` beginnings.

**3.4 Minimum length `min_words: 50` (step 3b).** Very short texts carry little stylistic signal for a detector and are a source of noisy labels. Counted as whitespace-separated words (after step 2). Effect on the full data: 104,260 -> 101,900 rows.

**3.5 Label conflicts removed (step 4).** If the same text appears both as human and as machine, the label is unreliable, so **all** copies are dropped rather than keeping one. In the current data this removed 0 rows; the step is a guard for future sources.

**3.6 Deduplication is global and exact (step 5), after normalisation.**
In M4 the same human text is repeated in every `<source>_<generator>` file of its domain (about 3000 unique human texts per source against ~15000 rows). Without dedup:
- classes would be artificially skewed towards the human side and repeated texts would be over-weighted,
- the same text could end up in train and test.

Dedup is on the normalised text, global (not per source), `keep="first"`. Near-duplicates (paraphrases, different truncation) are **not** detected. Effect: 101,900 -> 60,861 rows 

**3.7 Generator families (`family`).**
Raw names differ between files for the same system, so the model name is mapped to a family by prefix (`data.model_families`, matched case-insensitively with `startswith`):

| raw `model` | `family` |
|---|---|
| `cohere`, `cohere-xlarge-nightly`, `command-xlarge-nightly` | `cohere` |
| `gpt-3.5-turbo` | `chatgpt` |
| `text-davinci-003` | `davinci` |
| `dolly-v2-12b` | `dolly` |
| `flan-t5` | `flan-t5` |

Unknown names fall back to their lower-cased name (so a new generator becomes its own family instead of failing). `model` is kept untouched, `family` is used for balancing and for analyses per generator.

**3.8 Undersampling (step 6).**
After dedup the data is ~12k human vs. ~51k machine texts (≈ 1:4). Undersampling the majority class was chosen over duplicating human texts or loss weights.
Procedure (`undersample`):
1. **Per source** quota `per_class = min( smallest (source, label) group , max_samples // (2 * n_sources) )`. With the current data: `min(≈3000, 10000 // 8) = 1250`.
   Every source contributes equally and every source is balanced 50/50, so the label cannot be predicted from the domain alone.
2. Human texts: random sample of `per_class` per source.
3. Machine texts: `per_class` per source, spread **evenly across generator families** (`_sample_evenly`). If a family has too few texts, the shortfall is redistributed over the others, so the quota is always met when the source has enough machine texts.
4. Concatenate and shuffle.

Randomness uses `cfg["seed"]` (42) -> fully reproducible.
Result: 10,000 rows, 5,000 / 5,000.

**3.9 `max_samples` is a hard upper bound on the final size**, not a target applied before balancing, so changing it keeps classes and sources balanced.

### Resulting dataset (default config)

| source | human (0) | machine (1) |
|---|---|---|
| arxiv | 1250 | 1250 |
| reddit | 1250 | 1250 |
| wikihow | 1250 | 1250 |
| wikipedia | 1250 | 1250 |

Machine texts per family: `arxiv`, `reddit` -> 250 × 5 families; `wikihow`, `wikipedia` -> ~312 × 4 families (no `flan-t5` there).

Row counts through the pipeline: 106,066 loaded -> 101,900 cleaned (steps 1–3b) -> 101,900 after label-conflict removal -> 60,861 after dedup -> 10,000 balanced.

---

## 4. Splitting and saving

### `split(df, data_config, seed)` -> `{"train", "val", "test"}`

Stratified split with proportions `data.split` (default 70 / 15 / 15).

**4.1 Stratification key = (`source`, `family`)**, human texts being their own family (`"human"`). This single key fixes label, domain and generator-family shares in every split: each cell (e.g. `reddit` × `cohere`) is shuffled and cut separately, so train/val/test have the same composition as the balanced dataset. Stratifying only by label would let `family` or `source` proportions drift between splits and make val/test results depend on luck. 

**4.2 Split names live in the config.** The keys of `data.split` (`train`, `val`, `test`) define the names, their order and the output file names. Cut points are `round(n * cumulative fraction)`, and the last split takes the remainder of the cell, so no row is lost or duplicated.

**4.3 No text overlap between splits.** Texts are unique after `preprocess` (step 5), so splitting rows is safe. The function still asserts that the three text sets are disjoint.

**4.4 Reproducibility.** Shuffling uses `random_state = cfg["seed"]` (42), the same seed as sampling in `preprocess`.

**4.5 Config is validated**: `data.split` needs at least two entries and must sum to 1. Downstream stages (detector training/evaluation) expect the names `train`, `val`, `test`.

### `save_splits(splits, data_config)`

**4.6 Format: JSON Lines, one file per split:** `<data.dir>/train.jsonl`, `val.jsonl`, `test.jsonl`. JSONL matches the raw M4 format, is human-readable/greppable, streams well, and needs no extra dependency. `force_ascii=False` keeps non-ASCII characters readable.

**4.7 Columns are preserved:** `text`, `label` (0 human / 1 machine), `source`, `model`, `family`. `model`/`family`/`source` are kept so the detector and the explanation experiments can report results per domain and per generator.

**4.8 Existing files are overwritten**, the directory is created if missing.

**4.9 Fails early on empty input.** If no M4 records are found, `prepare_data` raises `FileNotFoundError`.

### Result (default config)

| split | rows | human | machine |
|---|---|---|---|
| train | 6998 | 3500 | 3498 |
| val | 1508 | 752 | 756 |
| test | 1494 | 748 | 746 |

### Tests

`tests/test_datasets.py` (synthetic data, no submodule needed): undersampling balance and cap, split exhaustiveness / disjointness / stratification, reproducibility, save->load round trip.
`tests/test_pipeline.py` stubs `prepare_data`, because CI does not fetch the ~2 GB submodule and the real stage writes to `data/`.

---

## 5. Configuration reference (`configs/default.yaml` -> `data:`)

Top-level `seed` (42) drives all random sampling and shuffling.

| Key | Default | Used in | Meaning |
|---|---|---|---|
| `name`, `language` | `M4`, `en` | `load_m4` (log only) | labels; the language is enforced by `domain`, not by these keys |
| `raw_dir` | `external/M4/data` | `load_m4` | input directory with `*.jsonl` |
| `dir` | `data/` | `save_splits` | output directory for the splits |
| `domain` | `[wikipedia, reddit, wikihow, arxiv]` | `load_m4` | sources to load; `peerread` left out |
| `excluded_models` | `[bloomz]` | `load_m4` | generators skipped because of prompt echo |
| `human_text_keys` | `[human_text, text, abstract]` | `load_m4` | human text key candidates, first non-empty wins |
| `machine_text_key` | `machine_text` | `load_m4` | key of the machine text |
| `model_families` | prefix -> family map | `load_m4` | generator families; add a prefix to support a new generator (no code change) |
| `strip_leading_regex` | `^[;,\s]+` | `preprocess` step 2b | leading separator junk (WikiHow artifact) |
| `prompt_prefixes` | `[Write, Please write, I will ask you]` | `preprocess` step 3 | prompt-echo filter for machine texts |
| `min_words` | `50` | `preprocess` step 3b | minimum text length in words |
| `max_samples` | `10000` | `undersample` | upper bound of the balanced dataset |
| `split` | `{train: .70, val: .15, test: .15}` | `split`, `save_splits` | split names (= file names), their order and proportions |

