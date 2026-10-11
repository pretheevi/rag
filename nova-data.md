# NOVA data to store in RAG

Source root: `NOVA_AUTO_Prithvi/` (all paths below are relative to it).
Docs and test cases come from Jira, so they are not listed here.
This file lists names and locations only. No file contents were copied.

## 1. API specs (highest value)
One OpenAPI yaml per feature. Index all of them, including `common/schemas.yaml` in each service.

| Service | Files in `api-spec/` |
|---|---|
| nova-practice | appointment-restrictions, appointment-statuses, appointment-types, appointment-zones, available-schedule-tabs, checkin-checkout-flow, default-location, display-preferences, locations, login-events, opevent-stream, patient-tracking-statuses, rooms, schedule-locations, schedule-tabs, schedule-zones, schedules, staff-locations, staff, user-profile, zone-purposes |
| nova-medical | clinical-sections, enc-templates, encounter-notes, encounters, physicals, ros, vitals, hello |
| nova-demographic | registers, hello |
| nova-documents | docimage-metadata, hello |
| nova-financial | archive-transactions, claims, hello |
| nova-messaging | esmg-deliveries, hello |

Pattern: `nova-<service>/api-spec/nova-<service>-<feature>.yaml` and `nova-<service>/api-spec/common/schemas.yaml`

## 2. Backend source code (Java)
Take `src/main/java` only.

- `nova-practice/practice-app`: config, domain, location, login, opevent, profile, schedule, schedulelocation, scheduling, staff, stafflocation
- `nova-medical/medical-app`: api, config, encnote, encounter, enctemplate, physical, ros, visibility, vitals, hello
- `nova-worker/worker-app`: config, login, tenant
- `nova-demographic/demographic-app`, `nova-documents/documents-app`, `nova-financial/financial-app`, `nova-messaging/messaging-app`
- `nova-shared-libraries/nova-infrastructure`: audit, cache, config, opevent, registry, resilience, security, tenant, time, transaction, web
- `nova-shared-libraries/nova-test-support` (test helper code)

Package root: `com.connexin.nova.<service>`

## 3. Database schema (SQL)
Take **schema** files only, from each `nova-<service>/docker/mysql/init/`:
- practice: 100-master, 120-tenant, 200-profile, 220-staff-schedule, 240-scheduling-settings, 260-code-table, 280-checkin-checkout-flow, 297-schedule-card-demographics
- medical: 600-enc-note, 620-physical, 640-template-tables, 660-encounter-tables, 670-visibility, 690-template-userprefs, 695-vitals-userprefs
- demographic, documents, financial, messaging, worker: the `*-schema.sql` files in their `docker/mysql/init/`

Skip `*-seed.sql`, `test/resources/**/*.sql` and `golden/*.sql`. They hold sample data, not structure.

## 4. Frontend source (React/TypeScript)
- `nova-web/src/`: domains (authentication, calendar, dashboard, encounter, locations, patient, users), components, routes, store, services, api, hooks, layout, config, lib
- `nova-web/README.md`, `nova-web/STATE_MANAGEMENT.md`
- `nova-web/package.json` (dependency names only)
- Skip `src/test`, `src/stories`, `src/assets`, `keycloak-theme`

## 5. QE automation framework (Playwright)
- `qe-pulse-2-0/src/`: pages, components, fixtures, utils, config, constants, enums, types, decorators, listeners, runner-manager, reporting, context, data, exceptions, annotations, auth (code only)
- `qe-pulse-2-0/tests/`: api, demographic, documents, financial, medical, messaging, practice (spec files, for example `practice/schedule-controller.spec.ts`, `medical/encounter-controller.spec.ts`, `*/session-security.spec.ts`, `api/nova-medical/NOVA-590.api.spec.ts`)
- `qe-pulse-2-0/README.md`, `qe-pulse-2-0/specs/README.md`, `playwright.config.ts`, `package.json`
- `qe-pulse-2-0/scripts/sync-runner-manager.ts`

## 6. Project docs and architecture
- README.md in: each `nova-*` root, `nova-*/docker/`, `nova-practice/docker/integration/`, `nova-worker/docker/integration/`, `nova-shared-libraries/*/`, `infrastructure/`, `infrastructure/cloudformation/observability/dashboards/`
- `nova-*/docker/test/*.sh` (API test scripts, for example `test-claims.sh`, `test-register.sh`)

## 7. Build, CI and infrastructure (optional, for deployment questions)
- `nova-*/pom.xml`
- `nova-*/.gitlab-ci.yml`, `infrastructure/.gitlab-ci.yml`
- `nova-*/ci/aws/deploys/**/*-template.yaml`
- `infrastructure/cloudformation/**/*.yaml`: alerts, compute (alb, ecs-cluster, elasticache-redis, sqs-audit-queue), frontend (s3-cloudfront), network (vpc, vpc-peering, private-dns), observability (grafana, mimir, tempo, collector, ...)
- `application.yaml` (base) per service: **after removing any secret, password, token or host values**

## 8. Sensitive items: record the name and location only, never the values

These are listed in RAG so users can find out they exist and where they live.
Store only: file name, path, service, purpose and the names of the keys or settings inside.
Never store: passwords, tokens, keys, certificate contents, connection strings, hostnames, or patient or sample data rows.

| Item | Location | What to record |
|---|---|---|
| Keystores and truststores (`keystore.p12`, `nova-keystore.p12`, `truststore.p12`) | every `*-app/src/main/resources` | Name, service, purpose. No file content. |
| Practice docker env file | `nova-practice/docker/nova-practice.env` | Variable names only |
| QE env files | `qe-pulse-2-0/env.dev`, `env.prod`, `env.qe`, `env.stag` | Variable names and which environment each is for |
| Web env files | `nova-web/.env.example`, `nova-web/.env.test` | Variable names only |
| Saved login sessions | `qe-pulse-2-0/.auth/` | Folder purpose only |
| Local and test app config | `application-local.yaml`, `application-test.yaml` per service | Property key names only |
| Credential handling code | `nova-shared-libraries/nova-infrastructure/.../registry/` (`AesCredentialDecryptor`, `CredentialDecryptor`, `CredentialDecryptionException`) | Class names and what they do; do not store any hardcoded key or secret if present |
| Seed and test SQL | `*-seed.sql`, `test/resources/**/*.sql`, `golden/*.sql` | File names and which tables they seed; no data rows |
| npm registry config | `nova-web/.npmrc` | Existence only |

Ingestion rule: before embedding any config or env file, strip every value and keep the keys.

## 9. Noise, no value
- `target/`, `node_modules/`, `dist/`, `build/`
- `qe-pulse-2-0/allure-report`, `allure-results`, `playwright-report`, `test-results`, `logs`
- `hs_err_pid*`, `*.mdmp`, `replay_pid*` (JVM crash dumps)
- `package-lock.json`, `yarn.lock`, `.png`, `.git`, `.husky`, `.vscode`, `.mvn`
- `NOVA-546.doc`, `NOVA-590.doc` and the test-case `.csv` files (come from Jira)

## Suggested metadata per chunk
`service`, `layer` (api-spec / java / sql / web / qe / infra), `file_path`, `feature` (for example schedules, encounters), `language`.
