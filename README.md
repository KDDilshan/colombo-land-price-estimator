# Colombo Land Price Estimator

[![tests](https://github.com/KDDilshan/colombo-land-price-estimator/actions/workflows/tests.yml/badge.svg)](https://github.com/KDDilshan/colombo-land-price-estimator/actions/workflows/tests.yml)

A tuned XGBoost model over 187 GN divisions in Colombo (50 features, including
seven environmental and spatial variables), with each GN's past flooding shown
from official Survey Department and DMC records, served as a SaaS product:
accounts, a free/Pro plan gated by Stripe, saved estimate history, PDF
reports, and a Pro-only "nearby amenities" lookup (schools, hospitals,
transport, etc. from OpenStreetMap).

Built as my BSc final-year thesis project.

## Model

| | |
|---|---|
| Model | Tuned XGBoost, predicting log(price per perch) |
| Data | 6,167 land listings across 187 GN divisions |
| Features | 50 (location, amenity distances/counts, environmental and flood hazard) |
| R² (80/20 held-out split, seed 42) | 90.11% |
| Median / mean absolute % error | 7.2% / 15.4% |

The app quotes one price per perch for each division and land type (the model
is asked for a standard 12-perch plot); the total is that price × the size you
enter. A tuned Random Forest (R² 89.46%) was served before the switch to
XGBoost.

The ML core (`predictor.py`, `deploy/`) is used exactly as trained — nothing
in the app retrains or refits it. The research pipeline that produced it is in
`research/` (see below).

## Local setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
copy .env.example .env         # then edit .env (blank Stripe/Google/mail is fine to start)

python wsgi.py
```

Open http://127.0.0.1:5000. With `.env` untouched (no `DATABASE_URL`, no
Stripe/Google keys), you get: SQLite at `./dev.db`, email+password signup and
login, a working free-tier estimator with quota enforcement — everything
except "Sign in with Google", Stripe checkout, and outgoing email (password
reset links print to the console instead).

Run the model's own tests any time with `pytest test_predictor.py` (after
`pip install -r requirements-dev.txt`) — these never touch the SaaS layer.
GitHub Actions runs them, plus an app start-up check, on every push and pull
request (`.github/workflows/tests.yml`).

## Wiring up the real integrations

**Stripe** (billing)
1. Create a Stripe account, switch to test mode.
2. Products → add a "Pro" product with a monthly recurring price → copy the price ID into `STRIPE_PRICE_PRO_MONTHLY`.
3. Developers → API keys → copy the secret/publishable keys.
4. Developers → Webhooks → add endpoint `<your-domain>/billing/webhook`, select `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted` → copy the signing secret into `STRIPE_WEBHOOK_SECRET`.
5. Test with card `4242 4242 4242 4242`, any future date/CVC.

Sri Lanka isn't a Stripe payout country, so the Stripe account needs to be
registered under a business entity Stripe does support (a supported country,
or a platform like Stripe Atlas) — that's a business decision outside this
codebase.

**Google sign-in**
1. Google Cloud Console → new project → APIs & Services → Credentials → OAuth client ID (Web application).
2. Authorized redirect URI: `<your-domain>/auth/login/google/callback`.
3. Put the client ID/secret in `.env`.

**Outgoing email** (password reset)
Any SMTP account works — a Gmail address with an
[app password](https://myaccount.google.com/apppasswords) is fine to start.
Set `MAIL_SERVER=smtp.gmail.com`, `MAIL_PORT=587`, `MAIL_USE_TLS=true`,
`MAIL_USERNAME`/`MAIL_PASSWORD` to the account, `MAIL_DEFAULT_SENDER` likewise.

## Deploying (Render)

1. Push this repo to GitHub.
2. Render → New → Blueprint → point at the repo (`render.yaml` is already set up: a web service + a free Postgres database).
3. Render will prompt for the `sync: false` env vars (Stripe, Google, mail) — paste in the real values.
4. Once live, update the Stripe webhook endpoint and the Google OAuth redirect URI to the Render URL.

Railway works the same way without `render.yaml` — create a service from the
repo, attach a Postgres plugin (sets `DATABASE_URL` automatically), set the
same env vars, and set the start command to `gunicorn -w 2 --timeout 60 wsgi:app`.

## Frontend styling (Tailwind)

The UI is Tailwind CSS compiled to a single static file
(`app/static/tailwind.css`), which is already built and committed — running
the app with `python wsgi.py` works with no Node/npm step at all. You only
need to rebuild it if you change a class in a template or edit
`app/static/src/input.css` (the design tokens / component styles):

```bash
npm install       # once
npm run build:css # rebuild app/static/tailwind.css after any template change
# or, while actively editing templates:
npm run watch:css # rebuilds automatically on every save
```

`app.js` and `site.js` are hand-written (no build step) and untouched by
this — they only depend on element IDs, which the redesign kept identical.

## Project layout

```
predictor.py, build_boundaries.py, deploy/                    the trained model + lookup tables
test_predictor.py                                              tests for the model only
research/                                                        the thesis pipeline (reference)
  src/step*.py                                                    data cleaning, geocoding, features, tuning, ablations
  scripts/                                                        Google Earth Engine flood/climate extraction
  notebooks/                                                      modelling notebooks
app/                                                             the Flask SaaS app
  config.py, extensions.py, models.py                           config, DB, User/EstimateHistory/AmenityCache
  amenities.py                                                    OpenStreetMap Overpass client + caching
  auth/, billing/, api/, dashboard/, admin/, main/                blueprints
  templates/, static/                                             pages, CSS, the map tool's JS
wsgi.py                                                          entrypoint (flask run / gunicorn)
```

The raw scraped listings and intermediate datasets are not included in this
repo, so the `research/` scripts are there to read, not to re-run as-is.

## Plans

Free: a limited number of estimates every 30 days (default 5, set by the
`FREE_TIER_MONTHLY_LIMIT` env var), no amenities, no PDF export. Pro:
unlimited estimates, amenities unlocked, full history, PDF export. Adjust the
Pro price/copy in `app/templates/pricing.html` to match your Stripe price.
