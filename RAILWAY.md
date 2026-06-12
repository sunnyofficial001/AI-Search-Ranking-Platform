# Production Deployment Guide: Railway Cloud Rollout

This manual maps out the deployment workflow for rolling out the **AI Search Ranking & Recommendation Platform** onto [Railway](https://railway.app/). 

This microservice topology utilizes a full-stack configuration consisting of a React-Express web server, a FastAPI ML microservice, PostgreSQL databases for telemetry queries, Redis caching, and integrated Elasticsearch retrieval.

---

## Architecture Topology Map

```
                     ┌───────────────────────┐
                     │     Railway Ingress   │
                     └───────────┬───────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │  ltr_web_frontend (3000)  │  <-- Serves React & SSR proxy
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │    ltr_api_service (8000) │  <-- FastAPI machine learning backend
                   └─────────────┬─────────────┘
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  PostgreSQL (DB) │   │   Redis Cache    │   │  Elasticsearch   │
├──────────────────┤   ├──────────────────┤   ├──────────────────┤
│ DB Query Logs    │   │ 136-dim Caching  │   │ Candidates Index │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## 1. Automated Provisioning checklist (One-click)

1. Open your **Railway Dashboard** and click **New Project** -> **Empty Project**.
2. Select **Provision PostgreSQL** to spin up a managed database instance.
3. Select **Provision Redis** to spin up a managed high-performance KV cache instance.

---

## 2. Deploying Python FastAPI API Service (`api_service`)

Configure the Python ML microservice to train RankNet and run candidate retrieval:

1. Click **New** -> **GitHub Repo** and choose this repository.
2. Under **Service Settings** -> **General Settings**:
   - Set **Service Name** to `api-service`.
   - Set the root Dockerfile to use the python default `Dockerfile`.
3. Go to the **Variables** tab and inject:
   - `PORT`: `8000`
   - `DATABASE_URL`: `${{Postgres.DATABASE_URL}}` *(Automatically references the provisioned PG Database)*
   - `REDIS_HOST`: `${{Redis.REDIS_HOST}}` *(Automatically references the provisioned Redis)*
   - `REDIS_PORT`: `${{Redis.REDIS_PORT}}`
   - `ELASTICSEARCH_HOSTS`: `http://your-elasticsearch-host:9200` *(Or integrate a cloud Elasticsearch addon)*
   - `MLFLOW_TRACKING_URI`: `http://your-mlflow-service`

---

## 3. Deploying React Node.js Frontend Service (`web_frontend`)

Deploy the frontend UI web app that proxies queries and queries models:

1. Click **New** -> **GitHub Repo** and choose this repository.
2. Under **Service Settings** -> **General Settings**:
   - Set **Service Name** to `web-frontend`.
   - Change the targeted Docker build recipe to `Dockerfile.frontend`.
3. In the **Variables** tab, inject connections:
   - `PORT`: `3000`
   - `NODE_ENV`: `production`
   - `API_URL`: `http://api-service.railway.internal:8000` *(Internal private networking dns)*
   - `GEMINI_API_KEY`: `${{Secrets.GEMINI_API_KEY}}` *(Passed securely for Gemini query expansion)*
4. Go to **Settings** -> **Public Networking** and click **Generate Domain** to map a secure HTTPS domain for the web app UI.

---

## 4. Initialization and Schema Migrations

Once PostgreSQL boots up:
- The `api_service` container automatically bootstraps and compiles tables via SQLAlchemy Declarative ORM Base mapping during engine initialization.
- If you wish to seed MOCK_PRODUCTS to PostgreSQL, trigger a shell instance through Railway CLI:
  ```bash
  railway run python -m backend.database.seed
  ```

---

## 5. Monitoring & Scalability

- **Cache hit ratios**: Utilize Redis CLI within the cache cluster to monitor keys matching `features:*` to verify feature extraction bypass workloads.
- **Auto-Scale rules**: Set replica scaling between 1 and 10 on the `api-service` based on CPU utilization crossing `70%`.
