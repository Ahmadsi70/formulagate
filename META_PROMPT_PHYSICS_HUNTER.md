# Meta-Prompt: Physics Formula Hunter Agent

> این prompt را به ابزار کدنویس (Cursor, Claude, ZCode, Copilot) بدهید تا پروژه را از صفر پیاده‌سازی کند.

---

## 📋 دستورالعمل پروژه

یک پروژه جدید پایتون به نام `physics-hunter` پیاده‌سازی کن. این پروژه یک Agent خودکار است که:

1. با Web Search API فرمول‌های فیزیک را از اینترنت جستجو و جمع‌آوری می‌کند
2. هر فرمول را از گیت FormulaGate عبور می‌دهد (Z3 proof verification)
3. نتایج (correct/incorrect/uncertain) را در PostgreSQL با pgvector ذخیره می‌کند
4. یک داشبورد Streamlit برای مشاهده و جستجوی نتایج دارد

---

## 🏗️ معماری و الزامات فنی

### ساختار پروژه
```
physics-hunter/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py            # تنظیمات + env vars
│   ├── crawler.py           # جستجوی وب + استخراج فرمول
│   ├── verifier.py          # اتصال به FormulaGate + اعتبارسنجی
│   ├── database.py          # PostgreSQL + pgvector
│   ├── scheduler.py         # زمان‌بندی خودکار (Celery / APScheduler)
│   ├── agent.py             # orchestrator اصلی — حلقه اصلی Agent
│   └── cli.py               # CLI برای اجرای دستی
├── dashboard/
│   └── app.py               # Streamlit dashboard
└── tests/
    ├── test_crawler.py
    ├── test_verifier.py
    └── test_database.py
```

### تکنولوژی‌ها
- **FormulaGate** (>=1.0.1): کتابخانه اصلی اعتبارسنجی — `from formulagate.sdk import Formulagate`
- **Brave Search API**: جستجوی وب (کلید رایگان دارد، $5/1000 query)
- **httpx**: HTTP client async برای crawl لینک‌ها
- **BeautifulSoup4**: استخراج متن از HTML
- **OpenAI API** (GPT-4o-mini): استخراج فرمول LaTeX از متن
- **psycopg2 + pgvector**: دیتابیس
- **APScheduler**: زمان‌بندی (سبک‌تر از Celery برای MVP)
- **Streamlit + Plotly**: داشبورد

---

## 🧩 جزییات پیاده‌سازی هر ماژول

### ۱. src/config.py
- خواندن از `.env`:
  - `BRAVE_API_KEY`
  - `OPENAI_API_KEY`
  - `DATABASE_URL` (PostgreSQL)
  - `CRAWL_INTERVAL_HOURS` (پیش‌فرض: 6)
  - `MAX_RESULTS_PER_QUERY` (پیش‌فرض: 20)
  - `LOG_LEVEL`

### ۲. src/crawler.py

کلاس `FormulaCrawler` با این متدها:

```python
class FormulaCrawler:
    def __init__(self, brave_api_key: str, openai_api_key: str)
    
    async def search_physics_formulas(
        self, 
        queries: list[str],      # کلمات کلیدی جستجو
        max_results: int = 20
    ) -> list[dict]
    # خروجی: [{"url": ..., "title": ..., "snippet": ..., "formulas_found": [...]}, ...]
    
    async def _search_brave(self, query: str, count: int) -> list[dict]
    # جستجو با Brave Search API — برگرداندن URL ها و snippet ها
    
    async def _fetch_page(self, url: str) -> str | None
    # دانلود صفحه با httpx (timeout=15s, respect robots.txt)
    
    async def _extract_formulas(self, text: str, url: str) -> list[dict]
    # استخراج فرمول‌های LaTeX از متن با GPT-4o-mini
    # prompt باید دقیقاً بگوید: فقط فرمول‌های فیزیک در قالب LaTeX برگردان
    # خروجی: [{"latex": "F = m a", "context": "Newton's second law..."}, ...]
```

**queries پیش‌فرض برای جستجو:**
```
"physics equation", "physics formula derivation",
"Newton's laws formula", "Maxwell equations", "Schrödinger equation",
"thermodynamics formula", "quantum mechanics equation",
"electromagnetism formula", "fluid dynamics equation",
"classical mechanics formula", "relativity equation",
"nuclear physics formula", "particle physics equation",
"statistical mechanics formula", "optics formula"
```

**استخراج فرمول با LLM — prompt دقیق:**
```
Extract ALL LaTeX physics formulas from the following text.
Return ONLY valid LaTeX expressions, one per line.
Do NOT include explanations, just the formulas.
Skip non-physics formulas (chemistry, math without physics context).
Format: each line = one LaTeX formula

Text:
{page_text}
```

### ۳. src/verifier.py

کلاس `FormulaVerifier` — wrapper دور FormulaGate:

```python
from formulagate.sdk import Formulagate, VerifyResult

class FormulaVerifier:
    def __init__(self, use_grounding: bool = True)
    
    def verify(self, formula: str, context: str = "") -> dict
    # خروجی:
    # {
    #   "formula": str,
    #   "parsed": bool,
    #   "ok": bool,
    #   "refuted": bool,
    #   "dimensions": str,
    #   "reason": str,
    #   "symbols": list[str],
    #   "structure_hash": str,
    #   "ground_truth": "correct" | "incorrect" | "uncertain",
    #   "label_confidence": float,
    # }
    
    def batch_verify(self, formulas: list[dict]) -> list[dict]
    # پردازش دسته‌ای — هر فرمول جداگانه verify می‌شود
    # فرمول‌های parse نشده (parsed=False) ذخیره نمی‌شوند
```

### ۴. src/database.py

کلاس `FormulaDatabase` — PostgreSQL + pgvector:

```sql
CREATE TABLE IF NOT EXISTS discovered_formulas (
    id              SERIAL PRIMARY KEY,
    formula         TEXT NOT NULL,
    latex           TEXT,
    dimensions      TEXT,
    symbols         TEXT[],
    structure_hash  TEXT UNIQUE,          -- dedup
    ground_truth    TEXT,                  -- correct/incorrect/uncertain
    label_confidence FLOAT,
    reason          TEXT,
    source_url      TEXT,
    source_title    TEXT,
    source_domain   TEXT,
    context_snippet TEXT,
    embedding       vector(384),          -- برای semantic search
    is_parsed       BOOLEAN DEFAULT FALSE,
    discovered_at   TIMESTAMPTZ DEFAULT NOW(),
    verified_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ground_truth ON discovered_formulas(ground_truth);
CREATE INDEX idx_source_domain ON discovered_formulas(source_domain);
```

متدهای اصلی:
```python
class FormulaDatabase:
    def save_formula(self, formula_data: dict) -> bool
    def formula_exists(self, structure_hash: str) -> bool
    def get_stats(self) -> dict
    def search_similar(self, query: str, limit: int = 10) -> list[dict]
    def export_dataset(self, ground_truth: str | None = None, limit: int = 10000) -> list[dict]
    def get_domains_distribution(self) -> list[dict]
```

### ۵. src/agent.py

کلاس `PhysicsHunterAgent` — orchestrator اصلی:

```python
class PhysicsHunterAgent:
    def __init__(self, config_path: str | None = None)
    
    async def run_once(self) -> dict
    # یک دور کامل:
    #   1. search → crawl → extract formulas
    #   2. dedup (بر اساس structure_hash)
    #   3. verify با FormulaGate
    #   4. save در database
    # خروجی: آمار این دور
    
    async def run_loop(self)
    # حلقه بی‌نهایت با APScheduler (هر N ساعت یکبار)
    
    def stats(self) -> dict
    # آمار کلی: تعداد کل، correct/incorrect/uncertain, domain distribution
```

**حلقه اصلی Agent (روش کار):**
```
1. برای هر query در لیست:
   a. Brave Search → بگیر top 10 results
   b. برای هر result:
      - fetch صفحه
      - extract فرمول‌ها با LLM
      - dedup با structure_hash
      - verify با FormulaGate
      - ذخیره در DB
   c. لاگ: چند فرمول جدید پیدا شد، چندتا correct/incorrect
2. sleep (CRAWL_INTERVAL_HOURS)
3. تکرار
```

### ۶. src/scheduler.py

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def setup_scheduler(agent: PhysicsHunterAgent, interval_hours: int = 6):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        agent.run_once,
        trigger="interval",
        hours=interval_hours,
        id="physics_hunt",
        replace_existing=True,
    )
    return scheduler
```

### ۷. src/cli.py

```python
# دستورات CLI:
#   python -m src.cli hunt          ← یک دور جستجو + verify
#   python -m src.cli daemon        ← اجرای مداوم با scheduler
#   python -m src.cli stats         ← آمار
#   python -m src.cli export        ← export dataset as JSON
#   python -m src.cli dashboard     ← اجرای Streamlit
```

### ۸. dashboard/app.py

Streamlit با این بخش‌ها:
- **Stats cards**: Total formulas, Correct, Incorrect, Uncertain
- **Domain pie chart**: توزیع بر اساس شاخه فیزیک
- **Timeline chart**: فرمول‌های جدید در طول زمان
- **Recent discoveries table**: آخرین فرمول‌های پیدا شده
- **Search bar**: جستجوی semantic در دیتابیس
- **Export button**: دانلود دیتاست

### ۹. requirements.txt
```
formulagate>=1.0.1
httpx>=0.27.0
beautifulsoup4>=4.12.0
openai>=1.0.0
psycopg2-binary>=2.9.0
pgvector>=0.3.0
apscheduler>=3.10.0
streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

---

## ⚠️ الزامات حیاتی (حتماً رعایت کن)

1. **FormulaGate کتابخانه است، نه fork.**
   - از PyPI نصب شود: `pip install formulagate>=1.0.1`
   - فقط API عمومی SDK استفاده شود: `Formulagate().verify()`
   - به internals (gate.py, dimensions.py) دسترسی مستقیم نداشته باشد

2. **Rate limiting و احترام به سرورها:**
   - بین هر crawl حداقل ۲ ثانیه delay
   - User-Agent معتبر: `PhysicsHunter/1.0 (research bot; contact@example.com)`
   - حداکثر ۵۰ صفحه در هر دور
   - کدهای 429 و 503 با exponential backoff مدیریت شوند

3. **Dedup قبل از verify:**
   - `structure_hash` تولید شده توسط FormulaGate معیار dedup است
   - اگر فرمولی قبلاً دیده شده، skip و فقط count افزایش یابد

4. **Fail gracefully:**
   - اگر FormulaGate نصب نباشد: هشدار + skip
   - اگر Brave API key نباشد: هشدار + exit
   - اگر PostgreSQL نباشد: fallback به SQLite
   - اگر OpenAI key نباشد: fallback به regex-only extraction

5. **لاگ‌گیری کامل:**
   - هر فرمول پیدا شده: `[INFO] Found: F = m a from https://...`
   - هر verify: `[INFO] Verified: CORRECT | F = m a`
   - آمار هر دور: `[INFO] Round #5: 143 new, 87 correct, 12 incorrect, 44 uncertain`

6. **تنظیمات از env می‌آیند، نه hardcode.**

---

## 🧪 تست‌ها

هر ماژول حداقل یک تست داشته باشد:
- `test_crawler.py`: mock Brave API + mock LLM
- `test_verifier.py`: mock FormulaGate verify
- `test_database.py`: تست با PostgreSQL تستی یا SQLite fallback

---

## 📦 تحویل نهایی

پروژه باید:
1. با `pip install -e ".[dev]"` قابل نصب باشد
2. با `python -m src.cli hunt` یک دور کامل اجرا شود
3. با `python -m src.cli dashboard` داشبورد باز شود
4. `.env.example` شامل تمام متغیرهای مورد نیاز باشد
5. `README.md` شامل: توضیح پروژه، نحوه نصب، نحوه اجرا، معماری