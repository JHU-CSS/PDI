"""
End-to-end OpenRouter prompting + persona prompting + metrics + heatmap.

- Robust OpenRouter chat-completions caller (429 Retry-After + exponential backoff + jitter)
- Concurrent prompting over a dataframe (ThreadPoolExecutor)
- Persona prompting (per-row system prompt)
- Parsing ratings (safer for 1–5 scales)
- Overall mean + CI, demographic means + CI, delta + CI, kappa
- Heatmap for |mean delta| by demographic across models
"""

import os
import re
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem, t
from sklearn.metrics import cohen_kappa_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Model Configuration ────────────────────────────────────────────────────────

OPENROUTER_MODELS = {
    "gpt-5.2":            "openai/gpt-5.2",
    "claude-sonnet-4.6":  "anthropic/claude-sonnet-4.6",
    "claude-opus-4.6":    "anthropic/claude-opus-4.6",
    "gemini-3.1-pro":     "google/gemini-3.1-pro-preview",
    "claude-haiku-4.5":   "anthropic/claude-haiku-4.5",
    "llama-3-8b":         "meta-llama/llama-3-8b-instruct",
    "mistral-large-2512": "mistralai/mistral-large-2512",
    "gpt-oss-120b":        "openai/gpt-oss-120b",
}

DEMOGRAPHICS = ["gender", "age", "education"]

DATASET_CONFIG = {
    "politeness": {
        "text_col":   "text",
        "rating_col": "politeness",      # human column name
        "id_col":     "instance_id",
        "scale_min":  1,
        "scale_max":  5,
    },
    "offensiveness": {
        "text_col":   "text",
        "rating_col": "offensiveness",   # human column name
        "id_col":     "user_id",
        "scale_min":  1,
        "scale_max":  5,
    },
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset CSV and return as DataFrame."""
    return pd.read_csv(path)

# ── OpenRouter API Call ────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(
    model_id: str,
    prompt: str,
    system: str | None = None,
    api_key: str | None = None,
    retries: int = 6,
    timeout_s: int = 60,
    max_tokens: int = 512,
    temperature: float = 0.0,
    app_url: str | None = None,
    app_title: str | None = None,
) -> str:
    """
    Call any model via OpenRouter Chat Completions endpoint with retries.

    - Handles 429 with Retry-After if present, otherwise exponential backoff + jitter
    - Raises helpful errors on non-200 responses
    """
    if not api_key:
        raise ValueError("Missing OpenRouter API key. Pass api_key=... or set OPENROUTER_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional but recommended attribution headers
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-OpenRouter-Title"] = app_title

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    sleep_s = float(retry_after)
                else:
                    sleep_s = (2 ** attempt) + random.random()
                time.sleep(sleep_s)
                continue

            if resp.status_code != 200:
                err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                # Don't retry permanent errors (e.g. 402 Insufficient Credits, 400, 401)
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise err
                raise err

            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_err = e
            time.sleep((2 ** attempt) + random.random())

    raise last_err if last_err else RuntimeError("Unknown OpenRouter error")

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_rating(response: str, scale_min: float = 1.0, scale_max: float = 5.0) -> float:
    """
    Extract a 1–5 style rating robustly.

    Accepts:
      - "4"
      - "4.0"
      - "Rating: 4"
      - "4/5"
      - "I would rate it a 4 out of 5"
    Avoids grabbing unrelated numbers when possible.
    """
    if not isinstance(response, str) or not response.strip():
        return np.nan

    # Prefer explicit "x/5" or "x out of 5"
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*5\b", response, flags=re.I)
    if m:
        x = float(m.group(1))
        return float(min(scale_max, max(scale_min, x)))

    # Otherwise, grab a standalone number in range
    # This pattern tries to avoid picking years etc by requiring word boundaries.
    candidates = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", response)
    for c in candidates:
        try:
            x = float(c)
            if scale_min <= x <= scale_max:
                return float(x)
        except ValueError:
            continue

    return np.nan

# ── Prompting (Concurrent) ────────────────────────────────────────────────────

def run_prompting(
    df: pd.DataFrame,
    dataset_name: str,
    build_prompt_fn,
    model_id: str,
    system: str | None = None,
    openrouter_api_key: str | None = None,
    max_rows: int | None = None,
    max_workers: int = 10,
    app_url: str | None = None,
    app_title: str | None = None,
) -> pd.DataFrame:
    """
    Run prompting over every row in df using concurrent OpenRouter API calls.

    Adds:
      - llm_text: raw model response
      - llm_rating: parsed numeric rating
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of: {list(DATASET_CONFIG.keys())}")

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
    else:
        df = df.copy()

    config = DATASET_CONFIG[dataset_name]
    id_col = config["id_col"]
    scale_min = float(config["scale_min"])
    scale_max = float(config["scale_max"])

    total = len(df)
    model_short = model_id.split("/")[-1].split(":")[0]

    llm_texts = [None] * total
    llm_ratings = [np.nan] * total

    rows = list(df.iterrows())

    def _call(idx: int, row: pd.Series):
        prompt = build_prompt_fn(row)
        text = call_openrouter(
            model_id=model_id,
            prompt=prompt,
            system=system,
            api_key=openrouter_api_key,
            app_url=app_url,
            app_title=app_title,
        )
        rating = parse_rating(text, scale_min=scale_min, scale_max=scale_max)
        return idx, text, rating

    print(f"  [{model_short}] submitting {total} calls (workers={max_workers})", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_call, i, row): i
            for i, (_, row) in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"  [{model_short}] {completed}/{total} done", flush=True)
            try:
                idx, text, rating = future.result()
                llm_texts[idx] = text
                llm_ratings[idx] = rating
            except Exception as e:
                orig_i = futures[future]
                row_id = rows[orig_i][1].get(id_col, orig_i)
                print(f"  ERROR on {id_col}={row_id}: {e}")

    success = sum(1 for r in llm_ratings if not (isinstance(r, float) and np.isnan(r)))
    print(f"  [{model_short}] done — {success} successful, {total - success} failed")

    df["llm_text"] = llm_texts
    df["llm_rating"] = llm_ratings
    return df

def run_persona_prompting(
    df: pd.DataFrame,
    dataset_name: str,
    build_prompt_fn,
    build_system_fn,
    model_id: str,
    openrouter_api_key: str | None = None,
    max_rows: int | None = None,
    max_workers: int = 10,
    app_url: str | None = None,
    app_title: str | None = None,
) -> pd.DataFrame:
    """
    Like run_prompting but with a per-row system prompt for persona experiments.

    Adds:
      - llm_text
      - llm_rating
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of: {list(DATASET_CONFIG.keys())}")

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
    else:
        df = df.copy()

    config = DATASET_CONFIG[dataset_name]
    id_col = config["id_col"]
    scale_min = float(config["scale_min"])
    scale_max = float(config["scale_max"])

    total = len(df)
    model_short = model_id.split("/")[-1].split(":")[0]

    llm_texts = [None] * total
    llm_ratings = [np.nan] * total

    rows = list(df.iterrows())

    def _call(idx: int, row: pd.Series):
        prompt = build_prompt_fn(row)
        system = build_system_fn(row)
        text = call_openrouter(
            model_id=model_id,
            prompt=prompt,
            system=system,
            api_key=openrouter_api_key,
            app_url=app_url,
            app_title=app_title,
        )
        rating = parse_rating(text, scale_min=scale_min, scale_max=scale_max)
        return idx, text, rating

    print(f"  [{model_short}] persona-prompting {total} rows (workers={max_workers})", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_call, i, row): i
            for i, (_, row) in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"  [{model_short}] {completed}/{total} done", flush=True)
            try:
                idx, text, rating = future.result()
                llm_texts[idx] = text
                llm_ratings[idx] = rating
            except Exception as e:
                orig_i = futures[future]
                row_id = rows[orig_i][1].get(id_col, orig_i)
                print(f"  ERROR on {id_col}={row_id}: {e}")

    success = sum(1 for r in llm_ratings if not (isinstance(r, float) and np.isnan(r)))
    print(f"  [{model_short}] done — {success} successful, {total - success} failed")

    df["llm_text"] = llm_texts
    df["llm_rating"] = llm_ratings
    return df

# ── Metrics ───────────────────────────────────────────────────────────────────

def _ci95(series: pd.Series) -> tuple[float, float]:
    """95% confidence interval using t-distribution."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n < 2:
        return (np.nan, np.nan)
    margin = t.ppf(0.975, df=n - 1) * sem(series)
    return (series.mean() - margin, series.mean() + margin)

def overall_mean(df: pd.DataFrame, rating_col: str = "llm_rating") -> dict:
    """Mean and 95% CI of LLM ratings across all rows."""
    ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
    lower, upper = _ci95(ratings)
    return {
        "mean":       round(float(ratings.mean()), 4) if len(ratings) else np.nan,
        "ci95_lower": round(float(lower), 4) if pd.notna(lower) else np.nan,
        "ci95_upper": round(float(upper), 4) if pd.notna(upper) else np.nan,
        "n":          int(len(ratings)),
    }

def demographic_mean(
    df: pd.DataFrame,
    demographic: str,
    rating_col: str = "llm_rating",
) -> pd.DataFrame:
    """Mean and 95% CI of LLM ratings grouped by a demographic."""
    results = []
    for group, subdf in df.groupby(demographic, dropna=False):
        ratings = pd.to_numeric(subdf[rating_col], errors="coerce").dropna()
        lower, upper = _ci95(ratings)
        results.append({
            demographic:  group,
            "mean":       round(float(ratings.mean()), 4) if len(ratings) else np.nan,
            "ci95_lower": round(float(lower), 4) if pd.notna(lower) else np.nan,
            "ci95_upper": round(float(upper), 4) if pd.notna(upper) else np.nan,
            "n":          int(len(ratings)),
        })
    return pd.DataFrame(results)

def delta(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str = "llm_rating",
) -> dict:
    """Mean delta (LLM - human) and 95% CI across all rows."""
    valid = df[[human_col, llm_col]].copy()
    valid[human_col] = pd.to_numeric(valid[human_col], errors="coerce")
    valid[llm_col] = pd.to_numeric(valid[llm_col], errors="coerce")
    valid = valid.dropna()
    diff = valid[llm_col] - valid[human_col]
    lower, upper = _ci95(diff)
    return {
        "mean_delta": round(float(diff.mean()), 4) if len(diff) else np.nan,
        "ci95_lower": round(float(lower), 4) if pd.notna(lower) else np.nan,
        "ci95_upper": round(float(upper), 4) if pd.notna(upper) else np.nan,
        "n":          int(len(diff)),
    }

def delta_by_demographic(
    df: pd.DataFrame,
    demographic: str,
    human_col: str,
    llm_col: str = "llm_rating",
) -> pd.DataFrame:
    """Mean delta (LLM - human) grouped by a demographic."""
    results = []
    for group, subdf in df.groupby(demographic, dropna=False):
        valid = subdf[[human_col, llm_col]].copy()
        valid[human_col] = pd.to_numeric(valid[human_col], errors="coerce")
        valid[llm_col] = pd.to_numeric(valid[llm_col], errors="coerce")
        valid = valid.dropna()
        diff = valid[llm_col] - valid[human_col]
        lower, upper = _ci95(diff)
        results.append({
            demographic:  group,
            "mean_delta": round(float(diff.mean()), 4) if len(diff) else np.nan,
            "ci95_lower": round(float(lower), 4) if pd.notna(lower) else np.nan,
            "ci95_upper": round(float(upper), 4) if pd.notna(upper) else np.nan,
            "n":          int(len(diff)),
        })
    return pd.DataFrame(results)

def cohen_kappa(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str = "llm_rating",
) -> float:
    """Cohen's kappa between LLM and human ratings (rounded to int)."""
    valid = df[[human_col, llm_col]].copy()
    valid[human_col] = pd.to_numeric(valid[human_col], errors="coerce")
    valid[llm_col] = pd.to_numeric(valid[llm_col], errors="coerce")
    valid = valid.dropna()

    if len(valid) == 0:
        return np.nan

    human_rounded = valid[human_col].round().astype(int)
    llm_rounded   = valid[llm_col].round().astype(int)
    return round(float(cohen_kappa_score(human_rounded, llm_rounded)), 4)

# ── Write LLM columns back to raw_data_llm.csv ───────────────────────────────

def write_llm_columns_back(
    dataset_name: str,
    duplicate_input_path: str,
    duplicate_output_path: str,
    all_model_labels: list[str],
    results_dict: dict[str, pd.DataFrame],
    column_suffix: str = "zero shot prompting",
) -> pd.DataFrame:
    """
    Merge per-model llm_rating columns back into the base raw_data_llm.csv.

    results_dict: {model_label: df_with_llm_rating} — in-memory results from run_prompting.
    Each model gets a column named: "{model_label} ({column_suffix})"
    Merge is done by id_col (instance_id / user_id), NOT by text.
    If a column already exists it is replaced (safe to re-run).
    """
    base_df = pd.read_csv(duplicate_input_path)

    for model_label in all_model_labels:
        if model_label not in results_dict:
            print(f"  [WARN] {model_label} not in results_dict, skipping")
            continue

        model_df = results_dict[model_label]
        col_name = f"{model_label} ({column_suffix})"

        # Index-based assignment — avoids cartesian product explosion when
        # id_col (e.g. user_id) is not unique per row.
        # results_dict values carry the original base_df index from run_prompting.
        col_series = pd.Series(np.nan, index=base_df.index)
        col_series.loc[model_df.index] = model_df["llm_rating"]

        if col_name in base_df.columns:
            base_df = base_df.drop(columns=[col_name])
        base_df[col_name] = col_series
        print(f"  [{dataset_name}] wrote column: {col_name!r}")

    base_df.to_csv(duplicate_output_path, index=False)
    print(f"  Wrote {len(all_model_labels)} model column(s) → {duplicate_output_path}")
    return base_df

# ── Visualization ─────────────────────────────────────────────────────────────

def plot_delta_heatmap(
    delta_dict: dict[str, pd.DataFrame],
    demographic: str,
    dataset_name: str,
    save_path: str | None = None,
):
    """
    Heatmap of |mean delta from human annotation| across models and demographic groups.

    delta_dict: {model_label: delta_by_demographic_df}
      where each df has columns: [demographic, mean_delta, ...]
    """
    if not delta_dict:
        raise ValueError("delta_dict is empty.")

    model_names = list(delta_dict.keys())
    first_df = next(iter(delta_dict.values()))
    groups = first_df[demographic].tolist()

    matrix = pd.DataFrame(index=groups, columns=model_names, dtype=float)
    for model_name, ddf in delta_dict.items():
        ddf_indexed = ddf.set_index(demographic)
        matrix[model_name] = ddf_indexed["mean_delta"].abs()

    fig, ax = plt.subplots(figsize=(len(model_names) * 2 + 2, max(4, len(groups) * 0.8)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "|Mean Delta from Human|"},
    )
    ax.set_title(
        f"|Delta from Human Annotation| by {demographic.capitalize()}\n"
        f"Dataset: {dataset_name}"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel(demographic.capitalize())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ── Example usage (edit to your paths/prompts) ────────────────────────────────

def example_build_prompt_politeness(row: pd.Series) -> str:
    return (
        "Rate the politeness of the following text on a 1-5 scale. "
        "Return ONLY a single number.\n\n"
        f"Text:\n{row['text']}"
    )

def example_build_prompt_offensiveness(row: pd.Series) -> str:
    return (
        "Rate the offensiveness of the following text on a 1-5 scale. "
        "Return ONLY a single number.\n\n"
        f"Text:\n{row['text']}"
    )

def example_build_persona_system(row: pd.Series) -> str:
    # Adjust to match your demographic fields
    gender = str(row.get("gender", "person"))
    age = str(row.get("age", "unknown age"))
    edu = str(row.get("education", "unknown education"))
    return (
        "You are a helpful rater. "
        f"Answer from the perspective of a {gender}, age {age}, education {edu}."
    )

def main_demo():
    # 1) Key
    api_key = os.environ.get("OPENROUTER_API_KEY")  # recommended
    # Or: api_key = "sk-or-..."

    # 2) Load data
    # df = load_dataset("politeness.csv")
    # dataset_name = "politeness"
    # build_prompt_fn = example_build_prompt_politeness

    df = load_dataset("offensiveness.csv")
    dataset_name = "offensiveness"
    build_prompt_fn = example_build_prompt_offensiveness

    # 3) Choose model
    model_id = OPENROUTER_MODELS["claude-sonnet-4.6"]

    # 4) Run prompting
    out = run_prompting(
        df=df,
        dataset_name=dataset_name,
        build_prompt_fn=build_prompt_fn,
        model_id=model_id,
        system=None,
        openrouter_api_key=api_key,
        max_rows=200,
        max_workers=10,
        app_title="project-multiperspective",
        app_url="https://example.com",
    )

    # 5) Metrics
    human_col = DATASET_CONFIG[dataset_name]["rating_col"]
    print("Overall mean:", overall_mean(out))
    print("Delta:", delta(out, human_col=human_col))
    print("Kappa:", cohen_kappa(out, human_col=human_col))

    # 6) Delta by demographic + heatmap (single model example)
    demo = "gender"
    ddf = delta_by_demographic(out, demographic=demo, human_col=human_col)
    print(ddf)

    plot_delta_heatmap(
        delta_dict={model_id: ddf},
        demographic=demo,
        dataset_name=dataset_name,
        save_path=None,
    )

if __name__ == "__main__":
    # main_demo()
    pass