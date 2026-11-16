# ACME CORPORATION — INFORMATION SECURITY & ACCEPTABLE USE POLICY
Document ID: DOC-SEC-2026-V2
Classification: Internal Restricted
Effective Date: February 15, 2026

## 1. Authentication and Password Controls
- **Password Length**: All directory and SSO passwords must contain a minimum of 16 characters.
- **Complexity**: Passwords must contain at least one uppercase letter, one lowercase letter, one numeric digit, and one special symbol (`!@#$%^&*()_+-=`).
- **Rotation Frequency**: Passwords must be changed every 90 days. Systems prevent the reuse of the previous 10 passwords.
- **Account Lockout**: After 5 consecutive failed login attempts, user accounts are locked for 30 minutes unless unlocked by IT SecOps.

## 2. Multi-Factor Authentication (MFA)
- Mandatory MFA is enforced on all corporate applications, email, GitHub repositories, and AWS/GCP administrative consoles.
- **Approved Authenticators**: Hardware security tokens (YubiKey 5 Series) or FIDO2/WebAuthn compliant authenticator apps (e.g., Google Authenticator, 1Password).
- **Prohibited Methods**: SMS-based two-factor authentication and email-based verification codes are strictly prohibited due to SIM-swapping and interception vulnerabilities.

## 3. Network Access & Virtual Private Network (VPN)
- All remote connections accessing production VPCs or internal staging environments must traverse the Zero-Trust Network Access (ZTNA) gateway with WireGuard/IPsec encryption.
- Direct public SSH or RDP access to any cloud instance is prohibited. Bastion hosts or AWS Systems Manager Session Manager must be utilized.
- Split-tunneling is strictly disabled on company-managed devices to prevent unauthorized egress.

## 4. Data Classification and Handling
- **Public**: Marketing assets, published documentation.
- **Internal**: General business communications, team wikis, organizational charts.
- **Confidential**: Source code, customer contact lists, financial forecasts. Encryption at rest (AES-256) and in transit (TLS 1.3) required.
- **Restricted**: PII, credit card data (PCI-DSS), cryptographic keys, production database credentials. Requires role-based access control (RBAC) and annual access re-certification.

## 5. Incident Response & Security Breaches
- All suspected security incidents, phishing attempts, lost devices, or unauthorized disclosures must be reported immediately to `security-incident@acme.corp` within 60 minutes of detection.
- Severity 1 (Critical) incidents initiate an automated bridge with the CISO, Legal Counsel, and Principal DevOps Engineer within 15 minutes of triage.
