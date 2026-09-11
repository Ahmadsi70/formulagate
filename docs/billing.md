# Billing & Subscription Guide

Formulagate uses a **Freemium** model: the core library is free and
open-source (Apache 2.0), while the hosted API offers tiered plans with
Dodo Payments-based self-service upgrades.

Dodo Payments acts as the **Merchant of Record** — it handles global tax
compliance (VAT, GST, sales tax), cross-border payments, and fraud
prevention so you don't need to register a business entity in every country.

---

## Plans

| Plan     | Requests / Month | Rate Limit / Minute | Features                                     | Price      |
|----------|:----------------:|:-------------------:|----------------------------------------------|:----------:|
| Free     | 1,000            | 10                  | `/verify`, `/check`, `/health`               | $0         |
| Pro      | 50,000           | 60                  | + `/db/*` access, exports, auto-labelling    | $49/mo     |
| Team     | 250,000          | 300                 | + priority routing                           | $299/mo    |
| Enterprise | Unlimited      | 1,000               | + on-premise, custom domain tables, SLA      | Contact us |

Upgrade paths available via Dodo Payments self-service: **Free → Pro**, **Pro → Team**.
Enterprise is sold through direct contact.

---

## Quick Start

### 1. Register and get an API key

```bash
curl -X POST https://api.formulagate.dev/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

Response (the API key appears **only once** — store it securely):

```json
{
  "api_key": "fg_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "api_key_prefix": "fg_live_a1b2c3d",
  "user_id": 42,
  "email": "you@example.com",
  "plan": "free",
  "created_at": "2026-08-30T12:00:00Z"
}
```

### 2. Use the API key

Pass it as the `X-API-Key` header:

```bash
curl https://api.formulagate.dev/verify \
  -H "X-API-Key: fg_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" \
  -H "Content-Type: application/json" \
  -d '{"formula": "E = m c^2"}'
```

### 3. Check your account

```bash
curl https://api.formulagate.dev/account \
  -H "X-API-Key: fg_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
```

Response:

```json
{
  "user_id": 42,
  "email": "you@example.com",
  "plan": "free",
  "plan_name": "Free",
  "monthly_quota": 1000,
  "usage_this_month": 15,
  "remaining": 985,
  "is_authenticated": true,
  "features": ["check", "health", "verify"],
  "rate_limit_per_minute": 10,
  "available_upgrades": ["pro", "team"],
  "subscription": null
}
```

### 4. Upgrade to Pro

```bash
curl -X POST https://api.formulagate.dev/billing/upgrade \
  -H "X-API-Key: fg_live_..." \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "pro",
    "success_url": "https://yourapp.com/upgrade-success",
    "cancel_url": "https://yourapp.com/upgrade-cancel"
  }'
```

Response includes a `checkout_url` — redirect the user there to complete payment
via Dodo Payments.

### 5. Manage your subscription

Users can manage their subscription through Dodo Payments' customer portal:

```bash
curl -X POST https://api.formulagate.dev/billing/portal \
  -H "X-API-Key: fg_live_..." \
  -H "Content-Type: application/json" \
  -d '{"return_url": "https://yourapp.com/account"}'
```

---

## API Authentication

### Auth is optional by default

Without setting any environment variable, the API behaves exactly as before:
all endpoints are public and no key is required. This is backward-compatible
with existing deployments.

### Enabling mandatory auth

Set `FORMULAGATE_REQUIRE_AUTH=true` to require an `X-API-Key` header on
every endpoint except `/health`, `/version`, `/`, `/auth/register`,
and `/billing/webhook`.

### Fail-open design

When PostgreSQL is unreachable and `FORMULAGATE_REQUIRE_AUTH` is **not**
set, requests fall back to anonymous (free-tier) access — the gate stays
available. When auth is required and the database is down, a 503 is returned.

---

## Quota & Rate Limiting

| Limit         | Free      | Pro       | Team      | Enterprise |
|---------------|:---------:|:---------:|:---------:|:----------:|
| Requests/month | 1,000    | 50,000    | 250,000   | Unlimited  |
| Requests/minute | 10      | 60        | 300       | 1,000      |

When the monthly quota is exhausted, the API returns **429 Too Many Requests**
with a message suggesting an upgrade. Rate limit headers are included in
every response:

```
X-RateLimit-Limit-Minute: 60
X-RateLimit-Remaining-Minute: 57
X-RateLimit-Limit-Hour: 10000
X-RateLimit-Remaining-Hour: 9995
```

---

## Dodo Payments Setup (Self-Hosted)

If you're running your own Formulagate API instance, set these environment
variables:

| Variable | Purpose |
|----------|---------|
| `FORMULAGATE_DODO_API_KEY` | Dodo Payments API key (from Developer → API Keys) |
| `FORMULAGATE_DODO_ENVIRONMENT` | `"test_mode"` or `"live_mode"` (default: `"live_mode"`) |
| `FORMULAGATE_DODO_WEBHOOK_SECRET` | Webhook signing secret (from Developer → Webhooks) |
| `FORMULAGATE_DODO_PRODUCT_PRO` | Product ID for the Pro subscription in Dodo |
| `FORMULAGATE_DODO_PRODUCT_TEAM` | Product ID for the Team subscription in Dodo |
| `FORMULAGATE_DATABASE_URL` | PostgreSQL connection string (required for billing) |

Install the billing extra:

```bash
pip install formulagate[billing,database,api]
```

### Create products in Dodo Payments dashboard

1. Go to **Products** in your Dodo Payments dashboard
2. Create a subscription product with the desired price ($49/month for Pro)
3. Copy its Product ID (starts with `pdt_`)
4. Set `FORMULAGATE_DODO_PRODUCT_PRO` to that ID
5. Repeat for Team ($299/month) and set `FORMULAGATE_DODO_PRODUCT_TEAM`

### Register the webhook

In Dodo Payments Dashboard → **Developer → Webhooks**, add:

```
https://your-domain.com/billing/webhook
```

Events to enable:
- `subscription.active`
- `subscription.updated`
- `subscription.cancelled`
- `payment.succeeded`
- `payment.failed`

Copy the webhook signing secret to `FORMULAGATE_DODO_WEBHOOK_SECRET`.

---

## Admin CLI

The CLI includes administrative commands for managing users and plans:

```bash
# Create an API key
formulagate admin key create --email user@example.com --plan pro

# List users
formulagate admin key list --limit 50

# Revoke a user
formulagate admin key revoke --email user@example.com

# Set a user's plan (override Dodo Payments)
formulagate admin plan set --user-id 42 --plan enterprise

# View a user's plan and usage
formulagate admin plan info --user-id 42

# View aggregate usage statistics
formulagate admin stats
```

---

## Security

- **API keys are never stored in plaintext** — only SHA256 hashes are persisted.
- The raw key is returned exactly once during registration and cannot be retrieved.
- Webhook signatures are verified using HMAC-SHA256.
- All API key usage is logged to an audit table (`api_key_audit`).
- Rate limiting operates at two layers: per-minute (middleware) and per-month (quota check).
- Dodo Payments handles PCI DSS Level 1 compliance and global tax collection.