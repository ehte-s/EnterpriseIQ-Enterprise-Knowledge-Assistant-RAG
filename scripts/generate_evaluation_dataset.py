"""Script to generate the comprehensive 100-question evaluation dataset for EnterpriseIQ."""

import json
from pathlib import Path

QUESTIONS = [
    # --- Category 1: Direct Factual Questions (25) ---
    {
        "id": "q_001",
        "category": "factual",
        "question": "What is the annual paid vacation leave entitlement for full-time employees?",
        "expected_answer": "Full-time employees are entitled to 25 days of paid annual vacation leave per calendar year.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "All full-time employees are entitled to 25 days of paid annual vacation leave per calendar year, accrued at 2.08 days per completed month of active service."
    },
    {
        "id": "q_002",
        "category": "factual",
        "question": "How many days of paid annual leave can be carried over into the next calendar year?",
        "expected_answer": "Up to 5 unused annual leave days may be carried over.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Up to 5 unused annual leave days may be carried over into the following calendar year, provided they are consumed prior to March 31."
    },
    {
        "id": "q_003",
        "category": "factual",
        "question": "What is the deadline by which carried-over annual leave days must be consumed?",
        "expected_answer": "Carried-over leave must be consumed prior to March 31.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "provided they are consumed prior to March 31. Any additional unused days expire without financial compensation."
    },
    {
        "id": "q_004",
        "category": "factual",
        "question": "How much advance notice is required for annual leave requests exceeding 3 consecutive days?",
        "expected_answer": "Requests must be submitted at least 14 calendar days in advance.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Leave requests exceeding 3 consecutive working days must be submitted through the Enterprise HR Portal at least 14 calendar days in advance"
    },
    {
        "id": "q_005",
        "category": "factual",
        "question": "How many paid sick leave days do full-time employees receive annually?",
        "expected_answer": "Employees receive 10 days of paid sick leave per calendar year.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Full-time employees receive 10 days of paid sick leave per calendar year, allocated on January 1."
    },
    {
        "id": "q_006",
        "category": "factual",
        "question": "When is an official medical certificate required for sick leave?",
        "expected_answer": "A medical certificate is required for absences lasting longer than 3 consecutive working days.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "For sick leave absences lasting longer than three (3) consecutive working days, an official medical certificate signed by a licensed medical practitioner must be uploaded"
    },
    {
        "id": "q_007",
        "category": "factual",
        "question": "Within what timeframe must a medical certificate be uploaded to the HR portal upon returning to work?",
        "expected_answer": "Within 48 hours of return to work.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "must be uploaded to the HR Portal within 48 hours of return to work."
    },
    {
        "id": "q_008",
        "category": "factual",
        "question": "How many weeks of fully paid parental leave are primary caregivers entitled to receive?",
        "expected_answer": "Primary caregivers are entitled to 16 consecutive weeks of fully paid parental leave at 100% base salary.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Primary Caregiver Leave: Any eligible employee who becomes a parent via birth, adoption, or surrogacy is entitled to 16 consecutive weeks of fully paid parental leave"
    },
    {
        "id": "q_009",
        "category": "factual",
        "question": "How many weeks of parental leave do secondary caregivers receive?",
        "expected_answer": "Secondary caregivers are entitled to 6 consecutive weeks of fully paid parental leave.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Secondary Caregiver Leave: Eligible secondary caregivers are entitled to 6 consecutive weeks of fully paid parental leave"
    },
    {
        "id": "q_010",
        "category": "factual",
        "question": "How many paid bereavement leave days are provided for immediate family members?",
        "expected_answer": "Up to 5 consecutive paid working days.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Employees are eligible for up to 5 consecutive paid working days for immediate family members (spouse, child, parent, sibling)"
    },
    {
        "id": "q_011",
        "category": "factual",
        "question": "What is the minimum character length required for company passwords?",
        "expected_answer": "All directory and SSO passwords must contain a minimum of 16 characters.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Password Length: All directory and SSO passwords must contain a minimum of 16 characters."
    },
    {
        "id": "q_012",
        "category": "factual",
        "question": "How frequently must passwords be rotated according to the IT Security policy?",
        "expected_answer": "Passwords must be changed every 90 days.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Rotation Frequency: Passwords must be changed every 90 days. Systems prevent the reuse of the previous 10 passwords."
    },
    {
        "id": "q_013",
        "category": "factual",
        "question": "How many previous passwords cannot be reused when changing a password?",
        "expected_answer": "The previous 10 passwords cannot be reused.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Systems prevent the reuse of the previous 10 passwords."
    },
    {
        "id": "q_014",
        "category": "factual",
        "question": "What happens after 5 consecutive failed login attempts?",
        "expected_answer": "User accounts are locked for 30 minutes.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "After 5 consecutive failed login attempts, user accounts are locked for 30 minutes unless unlocked by IT SecOps."
    },
    {
        "id": "q_015",
        "category": "factual",
        "question": "Which two multi-factor authentication methods are strictly prohibited?",
        "expected_answer": "SMS-based two-factor authentication and email-based verification codes are prohibited.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Prohibited Methods: SMS-based two-factor authentication and email-based verification codes are strictly prohibited"
    },
    {
        "id": "q_016",
        "category": "factual",
        "question": "What encryption standard is required for Confidential data at rest and in transit?",
        "expected_answer": "AES-256 for data at rest and TLS 1.3 for data in transit.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Confidential: Source code, customer contact lists, financial forecasts. Encryption at rest (AES-256) and in transit (TLS 1.3) required."
    },
    {
        "id": "q_017",
        "category": "factual",
        "question": "To what email address must suspected security incidents be reported?",
        "expected_answer": "They must be reported to security-incident@acme.corp.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "must be reported immediately to security-incident@acme.corp within 60 minutes of detection."
    },
    {
        "id": "q_018",
        "category": "factual",
        "question": "What protocol must internal synchronous communication between microservices use?",
        "expected_answer": "Internal synchronous communication must utilize gRPC with Protocol Buffers (proto3).",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Internal synchronous communication between microservices must utilize gRPC with Protocol Buffers (proto3)."
    },
    {
        "id": "q_019",
        "category": "factual",
        "question": "What messaging technology must event-driven communication use?",
        "expected_answer": "Apache Kafka with schema versioning in the Confluent Schema Registry.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Asynchronous Messaging: Event-driven communication must use Apache Kafka. Topic schemas must be versioned in the central Confluent Schema Registry."
    },
    {
        "id": "q_020",
        "category": "factual",
        "question": "What is the primary relational database store used by engineering?",
        "expected_answer": "PostgreSQL 16 managed via AWS RDS Aurora Multi-AZ.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Primary Relational Store: PostgreSQL 16 managed via AWS RDS Aurora Multi-AZ with read replicas"
    },
    {
        "id": "q_021",
        "category": "factual",
        "question": "What vector search engine is deployed for semantic retrieval in engineering architecture?",
        "expected_answer": "Qdrant cluster deployed with HNSW indexing and Cosine distance.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Vector Search Engine: Qdrant cluster deployed with HNSW indexing and Cosine distance metric for semantic retrieval."
    },
    {
        "id": "q_022",
        "category": "factual",
        "question": "Who can approve Tier 1 purchases under $1,000 USD?",
        "expected_answer": "The employee's direct line manager using a designated Corporate Purchasing Card.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Tier 1 Purchases (Under $1,000 USD): May be approved directly by the employee's direct line manager using a designated Corporate Purchasing Card (P-Card)."
    },
    {
        "id": "q_023",
        "category": "factual",
        "question": "What approvals are required for Tier 2 purchases between $1,000 and $10,000 USD?",
        "expected_answer": "Written approval from both the Department Director and the Financial Controller.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Tier 2 Purchases ($1,000 to $10,000 USD): Require written approval from both the Department Director and the Financial Controller"
    },
    {
        "id": "q_024",
        "category": "factual",
        "question": "What must be conducted for Tier 3 purchases exceeding $10,000 USD?",
        "expected_answer": "Competitive bidding with at least three qualified vendor quotes and CFO authorization.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Tier 3 Purchases (Exceeding $10,000 USD): Require competitive bidding with at least three (3) qualified vendor quotes and final authorization from the Chief Financial Officer (CFO)."
    },
    {
        "id": "q_025",
        "category": "factual",
        "question": "Within how many calendar days must expense reports be submitted?",
        "expected_answer": "Within 30 calendar days of incurring the expense.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Expense reports must be submitted via the Financial Cloud Portal within 30 calendar days of incurring the expense."
    },

    # --- Category 2: Numerical & Quantitative Questions (15) ---
    {
        "id": "q_026",
        "category": "numerical",
        "question": "What is the monthly accrual rate of annual vacation leave?",
        "expected_answer": "2.08 days per completed month of active service.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "accrued at 2.08 days per completed month of active service."
    },
    {
        "id": "q_027",
        "category": "numerical",
        "question": "On which day of absence does Short-Term Disability coverage begin and what percentage of salary is paid?",
        "expected_answer": "It begins on the 11th consecutive day of absence, paying 70% of base salary for up to 26 weeks.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "coverage beginning on the 11th consecutive day of medical absence, providing 70% of base salary for up to 26 weeks."
    },
    {
        "id": "q_028",
        "category": "numerical",
        "question": "What is the dollar amount of the one-time home office equipment stipend for remote employees?",
        "expected_answer": "$750 USD upon hire.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Remote employees receive a one-time home office equipment stipend of $750 upon hire"
    },
    {
        "id": "q_029",
        "category": "numerical",
        "question": "How much is the monthly internet and connectivity stipend for remote employees?",
        "expected_answer": "$60 USD per month.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "plus a recurring monthly internet and mobile connectivity stipend of $60."
    },
    {
        "id": "q_030",
        "category": "numerical",
        "question": "What are the weightings of company OKRs and individual performance in annual bonus calculations?",
        "expected_answer": "60% company OKR achievement and 40% individual performance score.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "weighted 60% company OKR achievement factor and 40% individual performance score."
    },
    {
        "id": "q_031",
        "category": "numerical",
        "question": "Within how many minutes must a security incident be reported after detection?",
        "expected_answer": "Within 60 minutes of detection.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "must be reported immediately to security-incident@acme.corp within 60 minutes of detection."
    },
    {
        "id": "q_032",
        "category": "numerical",
        "question": "How quickly must an automated bridge be initiated for Severity 1 security incidents?",
        "expected_answer": "Within 15 minutes of triage.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Severity 1 (Critical) incidents initiate an automated bridge with the CISO, Legal Counsel, and Principal DevOps Engineer within 15 minutes of triage."
    },
    {
        "id": "q_033",
        "category": "numerical",
        "question": "What is the core customer API uptime availability commitment and monthly downtime allowance?",
        "expected_answer": "99.95% uptime, translating to no more than 21.6 minutes of unplanned downtime per month.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Target Availability: The core customer-facing API tier maintains a minimum uptime commitment of 99.95% (three nines five), translating to no more than 21.6 minutes of unplanned downtime per calendar month."
    },
    {
        "id": "q_034",
        "category": "numerical",
        "question": "What are the P50, P95, and P99 API latency targets specified in engineering standards?",
        "expected_answer": "P50 < 80ms, P95 < 250ms, and P99 < 600ms.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "P50 latency < 80 milliseconds across all standard REST endpoints. P95 latency < 250 milliseconds. P99 latency < 600 milliseconds."
    },
    {
        "id": "q_035",
        "category": "numerical",
        "question": "How many consecutive failures within what time window trip a service circuit breaker?",
        "expected_answer": "5 consecutive failures within a 10-second window.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "configured to trip after 5 consecutive failures within a 10-second window."
    },
    {
        "id": "q_036",
        "category": "numerical",
        "question": "How many days are automated database backup snapshots retained, and what are the RPO and RTO targets?",
        "expected_answer": "Snapshots are retained for 35 days, with a 5-minute RPO and 1-hour RTO.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "snapshots retained for 35 days. Continuous point-in-time recovery (PITR) enabled with 5-minute RPO (Recovery Point Objective) and 1-hour RTO (Recovery Time Objective)."
    },
    {
        "id": "q_037",
        "category": "numerical",
        "question": "What is the minimum automated test code coverage threshold required for code merges to main?",
        "expected_answer": "At least 85% code coverage.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Mandatory code coverage threshold of at least 85%."
    },
    {
        "id": "q_038",
        "category": "numerical",
        "question": "What flight duration qualifies an employee to book business class air travel?",
        "expected_answer": "International flights exceeding 8 continuous flight hours with VP approval.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Business class airfare is permitted only for international flights exceeding 8 continuous flight hours with prior VP approval."
    },
    {
        "id": "q_039",
        "category": "numerical",
        "question": "What are the daily per diem meal caps for domestic and international travel?",
        "expected_answer": "$75 USD per day for domestic and $110 USD per day for international travel.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Domestic Travel: Up to $75 USD per day for meals. International Travel: Up to $110 USD per day for meals."
    },
    {
        "id": "q_040",
        "category": "numerical",
        "question": "Above what dollar amount are itemized receipts strictly mandatory for expense reimbursement?",
        "expected_answer": "Any individual expense exceeding $25 USD.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Itemized receipts are strictly mandatory for any individual expense exceeding $25 USD."
    },

    # --- Category 3: Multi-Document Questions (15) ---
    {
        "id": "q_041",
        "category": "multi_doc",
        "question": "What are the rules and requirements when working remotely regarding office stipends and network VPN security?",
        "expected_answer": "Remote employees receive a $750 equipment stipend and $60/month internet stipend, and must connect to the corporate VPN/ZTNA gateway when accessing internal services from public networks.",
        "relevant_document_ids": ["doc_hr", "doc_sec"],
        "expected_evidence": "Remote employees receive a one-time home office equipment stipend of $750... All remote connections accessing production VPCs or internal staging environments must traverse the Zero-Trust Network Access (ZTNA) gateway"
    },
    {
        "id": "q_042",
        "category": "multi_doc",
        "question": "What approvals and security reviews are required before purchasing a new SaaS software subscription?",
        "expected_answer": "Purchasing requires line manager or director approval depending on tier, plus mandatory IT Security review and Data Privacy impact assessment prior to purchase.",
        "relevant_document_ids": ["doc_fin", "doc_sec"],
        "expected_evidence": "Software & SaaS Licenses: Any software tool, cloud subscription, or AI API access must undergo IT Security review and Data Privacy impact assessment prior to purchase, regardless of dollar value."
    },
    {
        "id": "q_043",
        "category": "multi_doc",
        "question": "How is database access secured and what relational database engine is used in production?",
        "expected_answer": "PostgreSQL 16 Aurora Multi-AZ is used, and database credentials are classified as Restricted data requiring RBAC and encryption.",
        "relevant_document_ids": ["doc_eng", "doc_sec"],
        "expected_evidence": "PostgreSQL 16 managed via AWS RDS Aurora Multi-AZ... production database credentials. Requires role-based access control (RBAC)"
    },
    {
        "id": "q_044",
        "category": "multi_doc",
        "question": "What cache and vector technologies does engineering deploy, and how are travel expense reports managed in the cloud?",
        "expected_answer": "Engineering deploys Redis cluster (v7.2+) and Qdrant cluster, while travel expense reports are submitted via the Financial Cloud Portal.",
        "relevant_document_ids": ["doc_eng", "doc_fin"],
        "expected_evidence": "Redis cluster (v7.2+) used for session persistence... Vector Search Engine: Qdrant cluster... Expense reports must be submitted via the Financial Cloud Portal"
    },
    {
        "id": "q_045",
        "category": "multi_doc",
        "question": "If an employee has an extended medical leave, what disability benefits apply and how do performance reviews work?",
        "expected_answer": "Short-Term Disability provides 70% of base salary after day 10 for up to 26 weeks, while annual performance reviews occur in Q4 with adjustments taking effect February 1.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Short-Term Disability (STD) coverage beginning on the 11th consecutive day... Annual performance reviews occur during Q4"
    },
    {
        "id": "q_046",
        "category": "multi_doc",
        "question": "What authentication controls protect internal services, and what gRPC/REST protocols connect them?",
        "expected_answer": "Services are protected with mandatory MFA and ZTNA gateways, while internal microservices communicate via gRPC proto3 and external clients via REST/GraphQL.",
        "relevant_document_ids": ["doc_sec", "doc_eng"],
        "expected_evidence": "Mandatory MFA is enforced... Zero-Trust Network Access (ZTNA) gateway... internal synchronous communication between microservices must utilize gRPC with Protocol Buffers (proto3)."
    },
    {
        "id": "q_047",
        "category": "multi_doc",
        "question": "Compare the submission deadlines for sick leave medical certificates versus travel expense reports.",
        "expected_answer": "Medical certificates must be submitted within 48 hours of return to work, whereas travel expense reports must be submitted within 30 calendar days.",
        "relevant_document_ids": ["doc_hr", "doc_fin"],
        "expected_evidence": "must be uploaded to the HR Portal within 48 hours of return to work... Expense reports must be submitted via the Financial Cloud Portal within 30 calendar days"
    },
    {
        "id": "q_048",
        "category": "multi_doc",
        "question": "Which policies require VP-level approval?",
        "expected_answer": "Fully remote work arrangements require VP of People Operations approval, and international business class flights over 8 hours require VP approval.",
        "relevant_document_ids": ["doc_hr", "doc_fin"],
        "expected_evidence": "formal approval from the Department Head and the Vice President of People Operations... international flights exceeding 8 continuous flight hours with prior VP approval."
    },
    {
        "id": "q_049",
        "category": "multi_doc",
        "question": "What security procedures govern corporate laptops when employees work from home or hotel rooms?",
        "expected_answer": "Laptops must connect via ZTNA/VPN with split-tunneling disabled and MFA hardware tokens or approved authenticator apps.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "All corporate laptops must connect to the corporate VPN when accessing internal services... Split-tunneling is strictly disabled on company-managed devices"
    },
    {
        "id": "q_050",
        "category": "multi_doc",
        "question": "What are the rules regarding company credit cards and alcohol reimbursement during business travel?",
        "expected_answer": "Corporate cards can be used for authorized purchases, but alcohol is strictly non-reimbursable unless part of an approved client entertainment dinner.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "designated Corporate Purchasing Card (P-Card)... Alcohol is non-reimbursable unless associated with an approved client entertainment dinner."
    },
    {
        "id": "q_051",
        "category": "multi_doc",
        "question": "How are customer contact lists and source code classified, and what pipeline tests guard source code before deployment?",
        "expected_answer": "They are classified as Confidential requiring AES-256 and TLS 1.3, guarded by two senior code reviews, 100% test pass, 85% coverage, and Snyk/Trivy container scans.",
        "relevant_document_ids": ["doc_sec", "doc_eng"],
        "expected_evidence": "Confidential: Source code, customer contact lists... Branch protection with at least two senior engineer code reviews... Snyk and Trivy container scanning."
    },
    {
        "id": "q_052",
        "category": "multi_doc",
        "question": "What SLA uptime applies to the customer API, and who is paged during a Critical Severity 1 security outage?",
        "expected_answer": "99.95% uptime SLA, and Severity 1 incidents trigger an automated bridge with the CISO, Legal Counsel, and Principal DevOps Engineer within 15 minutes.",
        "relevant_document_ids": ["doc_eng", "doc_sec"],
        "expected_evidence": "Target Availability: The core customer-facing API tier maintains a minimum uptime commitment of 99.95%... automated bridge with the CISO, Legal Counsel, and Principal DevOps Engineer"
    },
    {
        "id": "q_053",
        "category": "multi_doc",
        "question": "What is the hybrid in-office attendance schedule, and does office equipment qualify for expense reimbursement?",
        "expected_answer": "Employees work on-site 3 days a week (Tue, Wed, Thu), while remote employees get a $750 home office equipment stipend upon hire.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "3:2 hybrid model: employees are required to work on-site... minimum of 3 days per week... home office equipment stipend of $750"
    },
    {
        "id": "q_054",
        "category": "multi_doc",
        "question": "Compare the consequences of 5 failed login attempts with 5 consecutive service client failures.",
        "expected_answer": "5 failed logins lock user accounts for 30 minutes, whereas 5 consecutive service failures trip a circuit breaker for 10 seconds.",
        "relevant_document_ids": ["doc_sec", "doc_eng"],
        "expected_evidence": "After 5 consecutive failed login attempts, user accounts are locked for 30 minutes... circuit breaker configured to trip after 5 consecutive failures within a 10-second window."
    },
    {
        "id": "q_055",
        "category": "multi_doc",
        "question": "What are the primary storage technologies used for relational data, session caching, and vector indexing across the company?",
        "expected_answer": "PostgreSQL 16 Aurora Multi-AZ for relational data, Redis cluster v7.2+ for cache/sessions, and Qdrant cluster for vector search.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "PostgreSQL 16 managed via AWS RDS Aurora Multi-AZ... Redis cluster (v7.2+)... Qdrant cluster deployed with HNSW indexing"
    },

    # --- Category 4: Contextual Understanding & Synthesis (15) ---
    {
        "id": "q_056",
        "category": "contextual",
        "question": "How does ACME's hybrid work policy structure collaborative office days?",
        "expected_answer": "It designates Tuesday, Wednesday, and Thursday as core on-site collaboration days under a 3:2 model.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Tuesday, Wednesday, and Thursday are designated core collaboration days"
    },
    {
        "id": "q_057",
        "category": "contextual",
        "question": "Why are SMS and email prohibited for multi-factor authentication?",
        "expected_answer": "Because of known SIM-swapping and message interception vulnerabilities.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "strictly prohibited due to SIM-swapping and interception vulnerabilities."
    },
    {
        "id": "q_058",
        "category": "contextual",
        "question": "How does the engineering deployment strategy mitigate risks during production releases?",
        "expected_answer": "By using a Canary deployment routing 10% of traffic for 15 minutes before 100% rollout, alongside automated test gates.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Canary deployment strategy: 10% traffic routing for 15 minutes before 100% progressive rollout."
    },
    {
        "id": "q_059",
        "category": "contextual",
        "question": "What measures prevent single point of failure in ACME's cloud infrastructure?",
        "expected_answer": "Multi-region redundancy across at least two independent AWS availability zones (us-east-1a and us-east-1b) with automatic failover.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Production services must be deployed across at least two independent AWS availability zones (us-east-1a and us-east-1b) with automatic health-check failover."
    },
    {
        "id": "q_060",
        "category": "contextual",
        "question": "What is the purpose of the competitive bidding requirement for purchases over $10,000?",
        "expected_answer": "To ensure financial prudence and value by requiring at least three qualified vendor quotes and CFO signoff.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Tier 3 Purchases (Exceeding $10,000 USD): Require competitive bidding with at least three (3) qualified vendor quotes and final authorization from the Chief Financial Officer (CFO)."
    },
    {
        "id": "q_061",
        "category": "contextual",
        "question": "How does the password policy prevent predictable password recycling by users?",
        "expected_answer": "By requiring 16 characters with complexity, rotating every 90 days, and blocking the reuse of the previous 10 passwords.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Passwords must be changed every 90 days. Systems prevent the reuse of the previous 10 passwords."
    },
    {
        "id": "q_062",
        "category": "contextual",
        "question": "What safeguards ensure customer data is protected during network transmission?",
        "expected_answer": "Zero-Trust Network Access (ZTNA) gateways with WireGuard/IPsec and TLS 1.3 encryption.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "traversal of the Zero-Trust Network Access (ZTNA) gateway with WireGuard/IPsec encryption... in transit (TLS 1.3) required."
    },
    {
        "id": "q_063",
        "category": "contextual",
        "question": "How does the organization balance domestic versus international travel expense allocations?",
        "expected_answer": "By providing higher caps for international travel ($110 per diem vs $75 domestic) and allowing business class only for international flights over 8 hours.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Domestic Travel: Up to $75 USD per day for meals. International Travel: Up to $110 USD per day for meals."
    },
    {
        "id": "q_064",
        "category": "contextual",
        "question": "What happens if an employee submits an expense report more than 60 days after incurring the expense?",
        "expected_answer": "Submissions exceeding 60 days are subject to outright rejection.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Late submissions exceeding 60 days are subject to rejection."
    },
    {
        "id": "q_065",
        "category": "contextual",
        "question": "How does the company handle parental leave for adoptive parents compared to biological parents?",
        "expected_answer": "Eligible employees becoming parents via birth, adoption, or surrogacy receive equal leave (16 weeks for primary, 6 weeks for secondary caregivers).",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Any eligible employee who becomes a parent via birth, adoption, or surrogacy is entitled to 16 consecutive weeks"
    },
    {
        "id": "q_066",
        "category": "contextual",
        "question": "What controls protect production cloud instances from direct public exposure?",
        "expected_answer": "Direct public SSH/RDP is prohibited; engineers must use bastion hosts or AWS Systems Manager Session Manager.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Direct public SSH or RDP access to any cloud instance is prohibited. Bastion hosts or AWS Systems Manager Session Manager must be utilized."
    },
    {
        "id": "q_067",
        "category": "contextual",
        "question": "What is the recovery objective if a catastrophic transactional database corruption occurs?",
        "expected_answer": "Point-in-time recovery with a 5-minute Recovery Point Objective (RPO) and 1-hour Recovery Time Objective (RTO).",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Continuous point-in-time recovery (PITR) enabled with 5-minute RPO (Recovery Point Objective) and 1-hour RTO (Recovery Time Objective)."
    },
    {
        "id": "q_068",
        "category": "contextual",
        "question": "How are security vulnerabilities detected before code reaches customer-facing environments?",
        "expected_answer": "Automated security scanning with Snyk and Trivy container scanning must find zero critical or high-severity vulnerabilities.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Zero critical or high-severity vulnerabilities identified by Snyk and Trivy container scanning."
    },
    {
        "id": "q_069",
        "category": "contextual",
        "question": "Under what circumstance does an employee forfeit accrued vacation days?",
        "expected_answer": "Carried-over days expire without compensation if not used by March 31, and any unused days beyond 5 cannot be carried over.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "provided they are consumed prior to March 31. Any additional unused days expire without financial compensation."
    },
    {
        "id": "q_070",
        "category": "contextual",
        "question": "How are hotel rates capped in high-cost cities like New York and Tokyo?",
        "expected_answer": "They are capped at $350 per night in high-cost cities, compared to $220 in standard metropolitan areas.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "standard corporate benchmark caps ($220/night in standard metropolitan areas; $350/night in high-cost cities such as New York, San Francisco, London, Tokyo, and Zurich)."
    },

    # --- Category 5: Questions with Adversarial / Distractor Passages (10) ---
    {
        "id": "q_071",
        "category": "distractor",
        "question": "Are employees allowed to book first class flights for long international journeys over 12 hours?",
        "expected_answer": "No. The policy only permits business class for international flights exceeding 8 hours with VP approval; first class is not permitted.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Business class airfare is permitted only for international flights exceeding 8 continuous flight hours with prior VP approval."
    },
    {
        "id": "q_072",
        "category": "distractor",
        "question": "Can an employee verify their login using an SMS text message if their phone app fails?",
        "expected_answer": "No. SMS-based authentication is strictly prohibited due to security vulnerabilities.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "SMS-based two-factor authentication and email-based verification codes are strictly prohibited"
    },
    {
        "id": "q_073",
        "category": "distractor",
        "question": "Can 10 days of unused annual vacation leave be carried over into next year if the manager agrees?",
        "expected_answer": "No. The policy strictly caps carry-over at a maximum of 5 days, which must be used by March 31.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Up to 5 unused annual leave days may be carried over into the following calendar year"
    },
    {
        "id": "q_074",
        "category": "distractor",
        "question": "Is mini-bar snacking at a corporate hotel reimbursable if within the daily meal per diem?",
        "expected_answer": "No. Hotel mini-bar purchases are explicitly classified as non-reimbursable personal expenses.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Non-Reimbursable Expenses: Personal grooming, traffic or parking violations, hotel mini-bar purchases"
    },
    {
        "id": "q_075",
        "category": "distractor",
        "question": "Can developers SSH directly to cloud production servers using their private SSH keys?",
        "expected_answer": "No. Direct public SSH access to cloud instances is prohibited; bastion hosts or AWS Systems Manager Session Manager must be used.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Direct public SSH or RDP access to any cloud instance is prohibited. Bastion hosts or AWS Systems Manager Session Manager must be utilized."
    },
    {
        "id": "q_076",
        "category": "distractor",
        "question": "Are part-time contractors entitled to the $750 home office equipment stipend?",
        "expected_answer": "No. The policy applies to full-time remote employees upon hire.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Remote employees receive a one-time home office equipment stipend of $750 upon hire"
    },
    {
        "id": "q_077",
        "category": "distractor",
        "question": "Can an employee submit a credit card summary slip instead of an itemized receipt for a $50 dinner?",
        "expected_answer": "No. Itemized receipts are strictly mandatory for expenses exceeding $25; credit card summary slips are invalid.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Itemized receipts are strictly mandatory for any individual expense exceeding $25 USD. Credit card summary slips without itemization are invalid."
    },
    {
        "id": "q_078",
        "category": "distractor",
        "question": "Can split-tunneling be enabled on personal laptops when accessing corporate resources?",
        "expected_answer": "Split-tunneling is strictly disabled to prevent unauthorized egress, and all traffic must pass through the ZTNA gateway.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Split-tunneling is strictly disabled on company-managed devices to prevent unauthorized egress."
    },
    {
        "id": "q_079",
        "category": "distractor",
        "question": "Does MongoDB or Cassandra serve as the primary relational database in the engineering architecture?",
        "expected_answer": "No. PostgreSQL 16 on AWS RDS Aurora Multi-AZ is the primary relational database.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Primary Relational Store: PostgreSQL 16 managed via AWS RDS Aurora Multi-AZ"
    },
    {
        "id": "q_080",
        "category": "distractor",
        "question": "Can an employee take 20 weeks of paid primary parental leave if twins are born?",
        "expected_answer": "No. The policy designates 16 consecutive weeks of fully paid parental leave for primary caregivers.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "entitled to 16 consecutive weeks of fully paid parental leave at 100% of regular base salary."
    },

    # --- Category 6: Questions with No Answer in Knowledge Base / Unanswerable (10) ---
    {
        "id": "q_081",
        "category": "unanswerable",
        "question": "What is the pet insurance subsidy provided by ACME Corporation?",
        "expected_answer": "The provided documents do not contain information regarding pet insurance subsidies.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_082",
        "category": "unanswerable",
        "question": "What discount do employees receive on corporate gym memberships?",
        "expected_answer": "The knowledge base does not mention corporate gym membership discounts.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_083",
        "category": "unanswerable",
        "question": "What are the rules regarding company stock option vesting upon retirement?",
        "expected_answer": "The available documents do not contain details regarding stock option vesting or retirement policies.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_084",
        "category": "unanswerable",
        "question": "What is the company policy on tuition reimbursement for master's degree programs?",
        "expected_answer": "The knowledge base does not contain any policy regarding educational tuition reimbursement.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_085",
        "category": "unanswerable",
        "question": "What brand of corporate laptops (MacBook Pro vs Dell XPS) is issued to new engineers?",
        "expected_answer": "The documents do not specify laptop hardware manufacturers or models.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_086",
        "category": "unanswerable",
        "question": "What is the corporate matching percentage for 401(k) retirement contributions?",
        "expected_answer": "The documents do not provide information on 401(k) retirement plans or company matching.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_087",
        "category": "unanswerable",
        "question": "How many days of paid volunteer time off (VTO) are employees granted each year?",
        "expected_answer": "The provided policies do not mention volunteer time off.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_088",
        "category": "unanswerable",
        "question": "What catering vendors are approved for on-site corporate team lunches?",
        "expected_answer": "The knowledge base contains no listing of approved catering vendors.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_089",
        "category": "unanswerable",
        "question": "What is the severance package formula in the event of workforce restructuring?",
        "expected_answer": "The knowledge base does not contain severance package terms or formulas.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },
    {
        "id": "q_090",
        "category": "unanswerable",
        "question": "What are the rules regarding sabbatical leave for employees with 5+ years of tenure?",
        "expected_answer": "The documents do not contain any policy governing sabbatical leave.",
        "relevant_document_ids": [],
        "expected_evidence": None
    },

    # --- Category 7: Questions Involving Document Metadata (10) ---
    {
        "id": "q_091",
        "category": "metadata",
        "question": "What is the document ID and version number of the Engineering Architecture Standards document?",
        "expected_answer": "Document ID: DOC-ENG-2026-V1, Version: 1.4.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Document ID: DOC-ENG-2026-V1\nVersion: 1.4"
    },
    {
        "id": "q_092",
        "category": "metadata",
        "question": "What is the document ID and effective date of the Global Human Resources & Workplace Policy?",
        "expected_answer": "Document ID: DOC-HR-2026-V3, Effective Date: January 1, 2026.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Document ID: DOC-HR-2026-V3\nEffective Date: January 1, 2026"
    },
    {
        "id": "q_093",
        "category": "metadata",
        "question": "What is the security classification and document ID of the Information Security & Acceptable Use Policy?",
        "expected_answer": "Document ID: DOC-SEC-2026-V2, Classification: Internal Restricted.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Document ID: DOC-SEC-2026-V2\nClassification: Internal Restricted"
    },
    {
        "id": "q_094",
        "category": "metadata",
        "question": "What is the document ID and effective date of the Procurement, Travel & Expense Policy?",
        "expected_answer": "Document ID: DOC-FIN-2026-V4, Effective Date: March 1, 2026.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Document ID: DOC-FIN-2026-V4\nEffective Date: March 1, 2026"
    },
    {
        "id": "q_095",
        "category": "metadata",
        "question": "What is the security classification of the Engineering Architecture Standards document?",
        "expected_answer": "Internal Confidential.",
        "relevant_document_ids": ["doc_eng"],
        "expected_evidence": "Classification: Internal Confidential"
    },
    {
        "id": "q_096",
        "category": "metadata",
        "question": "When does the Information Security & Acceptable Use Policy take effect?",
        "expected_answer": "Effective Date: February 15, 2026.",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Effective Date: February 15, 2026"
    },
    {
        "id": "q_097",
        "category": "metadata",
        "question": "What is the applicability scope stated in the Global Human Resources Policy?",
        "expected_answer": "All Full-Time and Part-Time Employees Globally.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Applicability: All Full-Time and Part-Time Employees Globally"
    },
    {
        "id": "q_098",
        "category": "metadata",
        "question": "What is the applicability scope of the Procurement, Travel & Expense Policy?",
        "expected_answer": "Global Enterprise Operations.",
        "relevant_document_ids": ["doc_fin"],
        "expected_evidence": "Applicability: Global Enterprise Operations"
    },
    {
        "id": "q_099",
        "category": "metadata",
        "question": "Which document has the classification 'Internal Restricted'?",
        "expected_answer": "The Information Security & Acceptable Use Policy (DOC-SEC-2026-V2).",
        "relevant_document_ids": ["doc_sec"],
        "expected_evidence": "Classification: Internal Restricted"
    },
    {
        "id": "q_100",
        "category": "metadata",
        "question": "Which enterprise policy document was effective earliest in the year 2026?",
        "expected_answer": "The Global Human Resources Policy, effective January 1, 2026.",
        "relevant_document_ids": ["doc_hr"],
        "expected_evidence": "Effective Date: January 1, 2026"
    }
]

def main():
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_questions.jsonl"
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in QUESTIONS:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully generated {len(QUESTIONS)} evaluation questions at: {out_file}")

if __name__ == "__main__":
    main()
