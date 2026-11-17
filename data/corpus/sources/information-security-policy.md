# Lumen Ledger Finance - Information Security Policy

> Fictional document created for the RAG-Ops demo corpus. Owner: Chief Information Security Officer (CISO). Version 4.2, effective 1 January 2026.

## 1. Purpose and Scope

This policy protects the confidentiality, integrity and availability of Lumen Ledger Finance information assets. It applies to all employees, contractors, interns and third parties who access company systems or data.

## 2. Data Classification

All information must be classified into one of four levels:

- **Public** - approved for external release, for example published interest rates and press releases.
- **Internal** - for employees only, for example internal policies and the org chart.
- **Confidential** - limited to specific teams, for example credit policy, compensation data and audit reports.
- **Restricted** - the most sensitive data, for example customer KYC documents, Aadhaar numbers, PAN, bank statements, passwords and encryption keys.

Restricted data must be encrypted at rest using AES-256 and in transit using TLS 1.2 or higher. Restricted data must never be stored on personal devices or personal cloud storage.

## 3. Access Control

Access is granted on the principle of least privilege. Access reviews are performed quarterly for systems holding Restricted data and semi-annually for all other systems.

When an employee exits, privileged access must be revoked within 4 hours and all other access within 24 hours of the last working day.

## 4. Passwords and Multi-Factor Authentication

Passwords must be at least 14 characters long. Passphrases are encouraged. Passwords are not force-rotated on a schedule; they must be changed immediately if a compromise is suspected.

Multi-factor authentication (MFA) is mandatory for email, VPN, the loan origination system (LOS) and all cloud consoles. FIDO2 hardware security keys are mandatory for production administrator accounts.

## 5. Endpoint Devices

All laptops must have full-disk encryption (BitLocker or FileVault) enabled. Screens must lock automatically after 5 minutes of inactivity.

USB mass-storage devices are disabled by default. An exception requires a ticket approved by the CISO, and the exception is valid for a maximum of 30 days.

## 6. Email, Messaging and AI Tools

Employees must not paste Confidential or Restricted data into public generative AI tools. The only approved AI assistant for company data is the internal assistant "Lumen Copilot". Customer data must never be shared over personal messaging apps.

## 7. Incident Reporting

Any suspected security incident must be reported to security@lumenledger.example or the SOC hotline at extension 7777 within 1 hour of discovery. Suspected phishing emails must be reported with the "Report Phish" button in the email client and must not be forwarded to colleagues.

## 8. Violations

Violations of this policy may lead to disciplinary action, up to and including termination of employment, and may be reported to law enforcement where required by law.
