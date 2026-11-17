# Lumen Ledger Finance - Incident Response Runbook

> Fictional document created for the RAG-Ops demo corpus. Audience: Engineering and Security. Owner: CTO and CISO.

## 1. Severity Levels

- **SEV1** - customer-facing outage of the loan origination system (LOS) or collections app, or a confirmed breach of Restricted data.
- **SEV2** - degraded performance affecting more than 25% of users, or a partial outage of a Tier-0 or Tier-1 system.
- **SEV3** - a minor issue with an available workaround.
- **SEV4** - a cosmetic issue with no functional impact.

## 2. Response Targets

| Severity | Acknowledge | Incident Commander assigned | Status updates | Executive notification |
| --- | --- | --- | --- | --- |
| SEV1 | 5 minutes | 15 minutes | Every 30 minutes | Within 30 minutes |
| SEV2 | 15 minutes | 30 minutes | Every 60 minutes | Within 2 hours |
| SEV3 | Next business day | Not required | Daily | Not required |

## 3. Incident Roles

- **Incident Commander (IC)** - owns the incident, makes decisions and declares resolution.
- **Operations Lead** - coordinates the technical investigation and mitigation.
- **Communications Lead** - sends stakeholder and customer updates using the approved templates.
- **Scribe** - maintains the incident timeline in the incident channel.

## 4. On-Call

On-call rotations are weekly, with the handoff every Monday at 10:00 IST. Pages are sent through the Lumen Pager system. If the primary on-call engineer does not acknowledge a page within 10 minutes, it escalates automatically to the secondary on-call engineer, and after a further 10 minutes to the engineering manager.

## 5. Recovery Objectives

| System tier | Example systems | RTO | RPO |
| --- | --- | --- | --- |
| Tier-0 | LOS, core lending ledger | 2 hours | 15 minutes |
| Tier-1 | Collections app, CRM | 8 hours | 1 hour |
| Tier-2 | Internal tools, HRMS | 24 hours | 24 hours |

Disaster recovery (DR) drills are conducted twice a year, in January and July.

## 6. Data Breach Procedure

For any suspected breach of Restricted data, the Incident Commander informs the CISO immediately, and the CISO informs the Data Protection Officer (DPO). Reportable cyber incidents must be notified to CERT-In within 6 hours of being noticed. Forensic images of affected systems must be preserved before remediation.

## 7. Postmortems

Postmortems are blameless. A postmortem is mandatory for every SEV1 and SEV2 incident and must be published within 5 business days of resolution. Every action item must have an owner and a due date and is tracked in Jira.
