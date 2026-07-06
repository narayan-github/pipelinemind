# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in PipelineMind, please report it
responsibly:

1. **Do not open a public issue.** Vulnerability details in a public tracker
   put all users at risk before a fix exists.
2. Email **narayan.samal@sigmoidanalytics.com** with:
   - A description of the vulnerability and its impact
   - Steps to reproduce (proof-of-concept if possible)
   - Any suggested mitigation or fix
3. You can also use GitHub's
   [private vulnerability reporting](https://github.com/narayan-github/pipelinemind/security/advisories/new)
   if enabled for this repository.

You should receive an acknowledgement within **72 hours**. Once the issue is
confirmed, we aim to release a fix as soon as practical and will credit you in
the changelog unless you prefer to remain anonymous.

## Scope & Common Areas of Concern

PipelineMind processes pipeline metadata and calls external LLM APIs. Reports
are especially welcome for:

- **Secret leakage** — API keys (e.g. `GROQ_API_KEY`) appearing in logs,
  traces, metrics, or LLM prompts
- **PII guard bypasses** — ways to make the API return data flagged in
  `data/catalogue/pii_registry.json` despite the PII middleware
- **Prompt injection** — crafted repository/document content that hijacks the
  agent's tool calls or approval flow
- **Unsafe tool execution** — agent tools performing actions outside their
  intended read-only/approved scope
- Standard web API issues (authn/authz, injection, SSRF) in the FastAPI service

## Best Practices for Deployers

- Never commit `.env` — it is gitignored for a reason; use `.env.example` as
  the template
- Rotate your Groq API key if you suspect exposure
- Change the default Grafana admin password (`docker-compose.yml`) before
  exposing monitoring beyond localhost
