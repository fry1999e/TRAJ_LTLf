# Traj-LTLf Framework: Natural Language to LTLf

A framework for translating natural-language temporal requirements into Linear Temporal Logic over finite traces (LTLf), combining source-grounded semantic construction, accessible human review, and trace consistency checks.

## 🌟 Overview

Traj-LTLf follows a **Review → Construct → Validate → Deliver** pipeline:

1. **Review**: Users review surface segmentation and atomic propositions (APs), while the primary generation model checks plain-language interpretations.
2. **Construct**: Build a semantic intermediate representation from the original requirement and confirmed APs, retaining interpretations that require further distinction.
3. **Validate**: Users judge whether finite scenarios satisfy the requirement using timeline tables; the framework performs structural, compilation, and consistency checks.
4. **Deliver**: Output one LTLf formula through normal selection or candidate fallback, preserving the review status and selection rationale.

### Key Features

- ✅ **Review for Nonexperts**: Users judge segmentation, AP meanings, and concrete scenarios without completing semantic structures, technical labels, confidence scores, or explanations.
- ✅ **Traceable AP Bindings**: Evidence is linked to original text spans; repeated predicates are reused, while overlapping evidence and compound predicates receive targeted repairs.
- ✅ **Adaptive Scenario Review**: Start with up to three key scenarios per group, review at most five distinct scenarios per requirement, and reuse judgments for identical timelines.
- ✅ **Single Formula Output**: When review remains incomplete, fallback can select from validated candidates while preserving raw human judgments and unapproved status.
- ✅ **Waiting and Request Optimization**: Display a spinner, reuse API connections, and overlap proposal requests with human input.
- ✅ **Batch Input and Resume**: Support requirement files, separate run directories, and batch checkpoint recovery.
- ✅ **Reproducibility Support**: Provide pinned dependencies, input digest verification, and complete review records.

---

## 🚀 Quick Start

### 1. Installation

Use Python 3.13. Open PowerShell in the project's parent directory, then create and activate a virtual environment:

```powershell
cd LTLf-main
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

`requirements.txt` lists direct dependencies; `requirements.lock.txt` pins the full dependency set.

### 2. Create `.env` File

Copy the template from the project root and configure the primary generation model's API:

```powershell
Copy-Item .env.example .env
```

```env
TRAJ_LTLF_LLM_MODE=api
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_model_name_here
LLM_API_BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0
```

The endpoint and model name must match your API service. See [.env.example](.env.example) for additional options. Existing process environment variables take precedence over values in `.env`.

### 3. Run the Framework

Run the human review workflow in an interactive terminal:

```powershell
python -B main.py --review manual --experiment-mode human --llm api --segmentation-review manual --plain-reading-review manual
```

Default settings are defined in [config/manual.json](config/manual.json). The default input is `data/input/example.xlsx`, and the default output directory is `data/output/manual/`. Running `python -B main.py` loads the same defaults. You can also provide a single natural-language requirement directly on the command line.

Follow the terminal prompts to complete the review:

- **Surface segmentation**: Enter `a` to accept, or enter `e` followed by the complete original text with only segmentation markers inserted.
- **APs**: Enter `a` to explicitly confirm, `e` to edit items interactively, or `e grant_occurs grant | occur` to modify only the specified AP. Enter `r` to regenerate or `c` to defer.
- **Scenario judgments**: Enter `y` (satisfies), `n` (does not satisfy), `u` (cannot determine), or `i` (infeasible scenario). Empty input does not grant approval.

The primary generation model handles plain-language interpretations and semantic structures; users provide only the three types of judgments above. The `--plain-reading-review manual` option is retained for compatibility. A spinner appears during waits and stops during human input. Use `--progress plain` for a static indicator or `--progress off` to disable it.

---

## 🧪 Reproducible Example

Append a single natural-language requirement to the command to run this example:

```powershell
python -B main.py --review manual --experiment-mode human --llm api --segmentation-review manual --plain-reading-review manual "Globally it is the case that if a holds then eventually a and b hold."
```

The intended reading here is: at every time step where `a` holds, `a` and `b` must hold simultaneously at the current or some later step. The transcript below preserves the review inputs, five scenarios, and judgment explanations from a complete interaction, omitting startup logs and waiting animations.

The prompts below are English translations of the terminal interaction; the current CLI may display them in Chinese. Different model services and prior judgments may change the number and order of scenarios and the final candidate. Judge the scenarios actually shown and retain the model configuration and review records.

<details>
<summary>Expand the full interaction: surface segmentation → AP confirmation → scenario judgments → formula output</summary>

### Surface Segmentation and AP Confirmation

First check that segmentation preserves the complete original text. This step confirms reading boundaries without requiring a technical semantic form:

```text
Original requirement:
Globally it is the case that if a holds then eventually a and b hold.

Complete original text with visible boundaries:
Globally it is the case that if a holds⟦CUT⟧ then eventually a and b hold.

Current reading segments:
  seg-1 [0:39] 'Globally it is the case that if a holds'
  seg-2 [39:69] ' then eventually a and b hold.'

Actions: a=accept; a!=accept and flag ambiguity; e/e! followed by the complete original text with only ⟦CUT⟧ inserted.
Surface segmentation review [1/3]: a

Please review the atomic propositions (APs) and their source evidence:
Globally it is the case that if a holds then eventually a and b hold.
  a: 'a' [32:33]; 'a' [56:57]
  b: 'b' [62:63]
AP review: a=explicitly confirm; r=regenerate; e=edit items; e name source_fragment1 | source_fragment2=edit only that AP; c=defer: a
```

The two occurrences of `a` denote the same atomic proposition; `b` is a separate proposition. Confirm these two APs. Their conjunction is represented later in the semantic structure.

### Judging Finite Scenarios

The scenario ends after the last row of each table. `true` means the proposition holds at that step; `false` means it does not. This interaction first confirms the requirement's meaning, then checks key scenarios, reusing judgments for identical timelines:

```text
Please judge whether each scenario in the tables satisfies the original requirement.
Original requirement: Globally it is the case that if a holds then eventually a and b hold.
true means the proposition holds at that step; false means it does not. The scenario ends after the last row.
Start with up to 3 key scenarios per group and add more only when needed; at most 5 distinct scenarios per requirement.
AP meanings:
  a: a holds
  b: b hold

Requirement meaning confirmation (0 distinct scenarios judged):

Key scenario 1/2 (total 1/5):
Time | a    | b
-----+------+-----
t0   | true | true
Choose: y satisfies / n does not satisfy / u cannot determine / i infeasible scenario: y

Key scenario 2/2 (total 2/5):
Time | a     | b
-----+-------+------
t0   | true  | false
t1   | false | true
Choose: y satisfies / n does not satisfy / u cannot determine / i infeasible scenario: n

Key scenario check (2 distinct scenarios judged):
Previous judgments for identical scenarios have been reused; no need to answer them again.

Key scenario 1/3 (total 3/5):
Time | a     | b
-----+-------+------
t0   | false | false
t1   | true  | false
Choose: y satisfies / n does not satisfy / u cannot determine / i infeasible scenario: n

Key scenario 2/3 (total 4/5):
Time | a     | b
-----+-------+------
t0   | true  | false
t1   | false | false
Choose: y satisfies / n does not satisfy / u cannot determine / i infeasible scenario: n

Key scenario 3/3 (total 5/5):
Time | a    | b
-----+------+------
t0   | true | false
t1   | true | true
Choose: y satisfies / n does not satisfy / u cannot determine / i infeasible scenario: y
```

These five judgments express the following:

| Scenario number | Input | Rationale |
|---|---|---|
| 1 | `y` | Both `a` and `b` hold at the current step, satisfying the reading that includes the current step. |
| 2 | `n` | Both `a` and `b` occur, but they never hold simultaneously. |
| 3 | `n` | `a` holds at the final step, with no joint occurrence satisfying the requirement before the scenario ends. |
| 4 | `n` | `a` triggers the obligation at t0, but no joint occurrence follows before the scenario ends. |
| 5 | `y` | The obligation from t0 is satisfied at t1; the new obligation at t1 is also satisfied at that same step. |

### Viewing the Formula and Review Status

After reaching the scenario limit, this run uses candidate fallback to output one formula. The terminal then shows the following, with variable elapsed-time prefixes omitted:

```text
The scenario review limit for this requirement has been reached. Completed judgments are retained; unfinished checks are handled by candidate fallback.
[RUN] [2] Writing outputs
[OK] Full result: data\output\manual\outputs\result.txt
[OK] Summary: data\output\manual\outputs\final_candidates.txt
[DONE] Traj-LTLf run finished
```

Open `data/output/manual/outputs/final_candidates.txt`. The final candidate for this run is:

```text
G(a -> F(a & b))
```

Here, `G` checks every time step, `->` expresses the conditional obligation, and `F(a & b)` requires `a` and `b` to hold simultaneously at the current or some future step. In comparison, `F a & F b` allows them to hold at different steps, which is why scenario 2 receives `n` in this example.

Runtime depends on API response times and human input speed and need not match the example. Candidate fallback preserves unfinished checks and the original approval status; obtaining a formula does not mean every gate has been approved. See `outputs/result.txt` for the complete result and `review_records/` in the same run directory for human answers and fallback evidence. Assign a separate `--run-dir` to each run when retaining multiple results. Online generation does not read offline reference answers.

</details>

---

## 📁 Project Structure

```text
LTLf-main/
├── main.py                          # CLI entry point
├── config/manual.json               # Default human review settings
├── src/                             # Core pipeline, agents, llm, rag, and contracts
├── data/
│   ├── input/                       # Runnable XLSX requirements and original sources
│   ├── examples/knowledge_base/     # General knowledge
│   ├── output/                      # Generated results and review records
│   ├── LTLf_Evaluation_Dataset.xlsx  # Complete requirements and offline reference formulas
│   ├── manifest.json               # Integrity digests for data and original sources
│   └── README.md                   # Data layout, fields, and usage
├── scripts/                         # Input exports and integrity checks
├── tests/                           # Automated tests
├── docs/                            # Architecture and detailed reproduction instructions
├── .env.example                     # API configuration template
├── requirements.txt                 # Direct dependencies
└── requirements.lock.txt            # Complete dependency lock
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for detailed usage and troubleshooting.

The `data` directory follows DSVA-Framework's `input / examples / output` layout, with the complete evaluation dataset stored directly under `data`. The datasets contain A72 (72 requirements), B240 (240), and C1026 (1,026), totaling 1,338 requirements. See [data/README.md](data/README.md) for directory and field descriptions. Files matching `input/*.xlsx` contain only `Natural Language` and `Requirement ID` columns; `input/sources/` retains the original source files for traceability. The complete evaluation workbook also includes `Raw Formula` and `Normalized Formula`, for offline comparison only.

For example, run the following to review A72:

```powershell
python -B main.py --review manual --experiment-mode human --llm api --segmentation-review manual --plain-reading-review manual --requirements-file data/input/A72.xlsx --run-dir data/output/A72
```

---
