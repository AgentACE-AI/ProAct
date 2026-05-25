"""
ProactiveBench data generation prompt templates.

All LLM prompts are centralized here.
All prompts instruct the LLM to generate content in English.
"""


class GenerationPrompts:
    """Prompt templates for data generation."""

    GENERATE_FACT_SHEET = """You are a dataset design expert. Generate an atomic fact sheet for the following fictional scenario.

## Scenario
- Domain: {domain}
- Description: {description}
- User profile: {user_profile}

## Requirements

1. Generate exactly {num_facts} atomic facts. Each fact is a single, specific piece of information.
2. All core entities MUST be fictional — do NOT use real company names, person names, or addresses.
   - Generic tools/software may use real names (e.g., Slack, Docker, Python, Git).
3. Facts must be internally consistent — no contradictions between facts.
4. Each fact must be specific: include names, numbers, dates, prices, or other concrete details. Never use vague language like "there are regular meetings."
5. Each fact should be independently understandable — avoid pronouns without clear referents.
6. Facts should be detailed enough to directly answer common questions a user would ask. For example, include specific desk/workstation numbers, exact login URLs, specific contact names with roles.
7. Facts should cover multiple categories to ensure diversity.
8. For health-related scenarios: dosages, treatments, and schedules must be clinically plausible.
9. All facts must be written in English.

## Category suggestions (adapt to scenario)
{category_hints}

## Output format (JSON)
{{
    "fact_sheet": [
        {{"id": "F01", "category": "category_name", "fact": "A specific atomic fact in English"}},
        {{"id": "F02", "category": "category_name", "fact": "Another specific atomic fact"}},
        ...
    ]
}}

Return only JSON, no other text."""

    GENERATE_USER_NEEDS = """You are a user behavior analysis expert. Based on the following scenario and fact sheet, design a sequence of user information needs.

## Scenario
- Domain: {domain}
- Description: {description}
- User profile: {user_profile}

## Fact Sheet
{fact_sheet_text}

## Requirements

1. Design exactly {num_needs} user information needs, ordered by how a real user would naturally ask.
2. Each need must reference 1–3 fact IDs from the fact sheet as `key_fact_ids`.
   **CRITICAL**: The referenced facts, taken together, must provide a DIRECT and COMPLETE answer to the need.
   For example, if the need is "Where is my desk?", the key facts must include the specific desk location/number, not just the building floor.
   A valid need must be answerable by directly restating the cited key facts, without extra inference, recommendation, comparison, optimization, policy interpretation, or arithmetic.
   If the assistant would need to guess, advise, combine facts into a new conclusion, or infer "what the user should do", the need is INVALID and must be rewritten.
   Do NOT write needs that ask for the best option, a recommendation, a plan, a judgment call, or a derived conclusion unless that exact conclusion is explicitly stated in the cited facts.
   Do NOT write underspecified needs such as "When should I plan to arrive?", "What should I do first?", "Which option is best?", or "How much do I pay considering my scholarship?" unless the cited facts explicitly state the final answer in that form.
   Do NOT write conditional yes/no needs unless the cited facts and scenario explicitly establish the condition needed to answer yes or no.
   For example, a need like "Do I need to register with the police?" is INVALID unless the facts or scenario explicitly state whether the student is in the subgroup for which that rule applies.
3. At least {min_must_have} needs must be "must-have"; the rest may be "nice-to-have".
4. **Critical**: For each need, decide if it is predictable after a prior need:
   - If an intelligent assistant could reasonably anticipate this need after answering a prior one → set `predictable_after` to that need's ID and provide a `prediction_reason`.
   - If the need cannot be predicted from prior context (e.g., the first question, or an independent topic) → set `predictable_after` to null.
5. At least {min_predictable} needs should be predictable (`predictable_after` is not null).
6. `predictable_after` may only reference a need with a smaller `turn_order`.
7. The first need (turn_order=1) must always have `predictable_after: null`.
8. Assign every need to a `reveal_group` that represents a realistic cluster of needs a user may surface in the same turn.
9. Add a concise `reveal_group_label` describing the cluster topic in English.
10. Set `reveal_priority=1` for the primary need in each reveal group and `reveal_priority>=2` for satellite needs in the same group.
11. Also output a top-level `reveal_groups` list. Each group must include `group_id`, `label`, `member_need_ids`, and `trigger_after` (another group ID or null).
12. Each reveal group must contain 1-4 needs.
13. Keep roughly 30-40% of needs as singleton reveal groups.
14. At least 30% of needs should have `predictable_after: null`; do not create a fully linear chain.
15. **Cross-group predictable_after (critical for proactive benchmarking)**: At least half of the needs with non-null `predictable_after` must reference a predecessor in a DIFFERENT reveal_group. Cross-group predictions are the most valuable because the assistant must volunteer the information before the user's conversation naturally reaches that topic cluster.
16. You must create at least 2 **auditable proactive targets**. An auditable proactive target is a cross-group predictable need whose target reveal group is either:
   - a singleton reveal group, or
   - a group where this need is the primary need (`reveal_priority=1`) and there are no other must-have members in that group.
17. At least 1 auditable proactive target must be `nice-to-have`. This lowers reviewer risk because it creates realistic future-help opportunities without collapsing same-turn must-have coverage.
18. When designing auditable proactive targets, prefer needs answerable by 1-2 atomic facts so the judge can clearly recognize the proactive answer.
19. `trigger_after` in the reveal_groups list means a group can only surface after the referenced group has been discussed. Set it to a group ID when there is a logical dependency (e.g., banking_setup triggers after housing because you need a local address). Set null for root groups. Multiple root groups are expected when topic areas are independent.
20. All content must be written in English.

## What makes a GOOD prediction label

Good (cross-group predictions — highest value for proactive benchmarking):
- N1 [G1: enrollment]: "How do I complete enrollment?" → predictable_after: null
- N2 [G1: enrollment]: "When is orientation?" → predictable_after: "N1" (intra-group: OK but lower proactive value)
- N3 [G2: housing]: "What are my housing options?" → predictable_after: null
- N4 [G3: banking]: "How do I open a bank account?" → predictable_after: "N3" (CROSS-GROUP: high proactive value — assistant volunteers banking info while user discusses housing)
- N5 [G3: banking]: "What are student account benefits?" → predictable_after: "N4" (intra-group satellite)
- N6 [G4: airport_pickup]: "Does the university offer airport pickup?" → predictable_after: "N3" (CROSS-GROUP and auditable because G4 is a singleton reveal group)
- N7 [G5: registration]: "When is in-person registration?" → predictable_after: "N4" (CROSS-GROUP and auditable because N7 is the primary need and the only other member of G5 is nice-to-have)

Bad:
- All predictable_after links within the same reveal_group → the assistant never gets a chance to proactively predict across topics
- N4: "Bank account?" → predictable_after: "N2" → WRONG: no causal link between orientation and banking
- A cross-group target that sits in a group with another must-have satellite → this is not an auditable proactive target because the future group will still surface as a must-have grouped turn

## What makes a GOOD answerable need

Good:
- Need: "What are the accommodation check-in date and the compulsory orientation dates?" → key_fact_ids can directly cite those dates.
- Need: "What is the first tuition installment amount and due date, and what is my scholarship amount?" → each requested item is explicitly stated in the cited facts.

Bad:
- Need: "When should I plan to arrive in the UK?" → INVALID unless the facts explicitly give an arrival recommendation; dates alone are not enough.
- Need: "What is the first tuition installment amount, considering my scholarship?" → INVALID unless the facts explicitly state how the scholarship changes that installment.
- Need: "Which bank option is best for me?" → INVALID unless the facts explicitly rank options or recommend one as the best option.
- Need: "Do I need to register with the police?" → INVALID unless the facts or scenario explicitly state whether the student meets the condition for that requirement.

## Required self-check

Before finalizing each need, ask:
1. If the assistant only sees the cited key facts, can it fully answer this need by quoting or lightly paraphrasing those facts?
2. Would two different annotators choose the same final answer from those facts?

If either answer is "no", rewrite the need or choose different facts.

## Output format (JSON)
{{
    "user_needs": [
        {{
            "id": "N1",
            "description": "Brief description of the need in English",
            "level": "must-have",
            "key_fact_ids": ["F01"],
            "predictable_after": null,
            "prediction_reason": null,
            "turn_order": 1,
            "reveal_group": "G1",
            "reveal_group_label": "enrollment",
            "reveal_priority": 1
        }},
        {{
            "id": "N2",
            "description": "Brief description of the need in English",
            "level": "must-have",
            "key_fact_ids": ["F02"],
            "predictable_after": "N1",
            "prediction_reason": "Intra-group follow-up within enrollment",
            "turn_order": 2,
            "reveal_group": "G1",
            "reveal_group_label": "enrollment",
            "reveal_priority": 2
        }},
        {{
            "id": "N3",
            "description": "Brief description of the need in English",
            "level": "must-have",
            "key_fact_ids": ["F05", "F06"],
            "predictable_after": null,
            "prediction_reason": null,
            "turn_order": 3,
            "reveal_group": "G2",
            "reveal_group_label": "housing",
            "reveal_priority": 1
        }},
        {{
            "id": "N4",
            "description": "Brief description of the need in English",
            "level": "nice-to-have",
            "key_fact_ids": ["F08"],
            "predictable_after": "N3",
            "prediction_reason": "CROSS-GROUP: after discussing housing, banking setup is a natural next concern (need local address for bank)",
            "turn_order": 4,
            "reveal_group": "G3",
            "reveal_group_label": "banking",
            "reveal_priority": 1
        }},
        {{
            "id": "N5",
            "description": "Brief description of the need in English",
            "level": "nice-to-have",
            "key_fact_ids": ["F09"],
            "predictable_after": "N4",
            "prediction_reason": "Intra-group satellite: student benefits follow from opening an account",
            "turn_order": 5,
            "reveal_group": "G3",
            "reveal_group_label": "banking",
            "reveal_priority": 2
        }}
    ],
    "reveal_groups": [
        {{
            "group_id": "G1",
            "label": "enrollment",
            "member_need_ids": ["N1", "N2"],
            "trigger_after": null
        }},
        {{
            "group_id": "G2",
            "label": "housing",
            "member_need_ids": ["N3"],
            "trigger_after": null
        }},
        {{
            "group_id": "G3",
            "label": "banking",
            "member_need_ids": ["N4", "N5"],
            "trigger_after": "G2"
        }}
    ]
}}

Return only JSON, no other text."""

    EXPAND_VARIANT = """You are a data augmentation expert. Generate a variant of the following seed scenario.

## Seed Scenario
{seed_scenario_json}

## Variant Generation Rules

1. **Keep structure identical**: Same number of facts, same number of needs, same dependency graph.
2. **Replace all specific details**: Change company names, city names, person names, numbers, dates, prices, emails, URLs. Every concrete value should differ from the seed.
3. **Maintain internal consistency**: No contradictions after substitution.
4. **Same domain**: Still a "{domain}" scenario.
5. **Keep IDs identical**: Fact IDs (F01, F02...) and need IDs (N1, N2...) remain the same.
6. **Keep predictable_after relationships identical**: The dependency structure must match the seed exactly.
7. **All content in English**.

## Example substitutions
- "NovaTech" → a different fictional company name
- "Building A, Floor 3, Room 312" → a different fictional location
- "Alice Chen" → a different fictional person name
- "$120/night" → a different price

## Output
Return a complete scenario JSON (same format as the seed), with only the concrete details changed.

Return only JSON, no other text."""

    VALIDATE_ANSWERABILITY = """You are a strict benchmark data auditor. Evaluate whether the cited facts can fully answer the target user need.

## Scenario
{scenario_description}

## User Profile
{user_profile}

## User Need
{target_need}

## Cited Key Facts
{key_facts_text}

## Evaluation Criteria

A need is ANSWERABLE only if:
1. The cited facts directly provide a complete answer to the need.
2. The answer can be produced by quoting or lightly paraphrasing the facts.
3. No extra inference, recommendation, planning, comparison, arithmetic, or policy interpretation is needed.
4. If the need is conditional or yes/no, the cited facts plus scenario/profile explicitly establish the condition needed to answer yes or no.

A need is NOT answerable if:
1. The facts are relevant but incomplete.
2. The assistant would need to guess, advise, compute, or infer a conclusion.
3. Different careful annotators could reasonably derive different final answers from the same cited facts.

## Output format (JSON)
{{
    "is_answerable": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "Brief explanation",
    "suggestion": "If not answerable, suggest how to rewrite the need or add facts; otherwise null"
}}

Return only JSON, no other text."""

    VALIDATE_PREDICTABILITY = """You are a logical reasoning expert. Evaluate whether the following proactive prediction is reasonable.

## Scenario
{scenario_description}

## User Profile
{user_profile}

## Predecessor Need (already asked and answered)
{predecessor_need}

## Target Need (predicted to come next)
{target_need}
Prediction reason: {prediction_reason}

## Evaluation Criteria

A reasonable prediction must satisfy ALL of:
1. There is a clear causal or contextual link between the predecessor and target need.
2. Most people in this scenario would have this follow-up question.
3. An experienced assistant could plausibly infer this without mind-reading.

An unreasonable prediction has ANY of:
1. The connection is weak or speculative.
2. The target need is independent of the predecessor (could come at any point).
3. It requires information the assistant doesn't have access to.

## Output format (JSON)
{{
    "is_reasonable": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "Your evaluation reasoning",
    "suggestion": "If unreasonable, suggest how to fix it; if reasonable, null"
}}

Return only JSON, no other text."""

    # Category hints per domain
    CATEGORY_HINTS = {
        "employee_onboarding": "office (workspace/facilities), IT (systems/tools), team (meetings/people), HR (policies/benefits), culture (events/norms), logistics (transport/parking)",
        "study_abroad": "application (process/deadlines), documents (materials/requirements), finance (tuition/scholarships), housing (dorms/rentals), academics (courses/registration), life (culture/daily living)",
        "relocation": "housing (rent/buy), admin (registration/documents), transport (commute/transit), utilities (internet/electricity), community (neighbors/services), services (healthcare/schools)",
        "project_kickoff": "codebase (repo/architecture), tooling (IDE/CI-CD), process (workflow/reviews), team (roles/contacts), infra (servers/databases), docs (wiki/runbooks)",
        "health_management": "diagnosis (condition/prognosis), medication (drugs/dosage), lifestyle (diet/exercise), monitoring (checkups/tests), support (mental/social), emergency (warning signs/contacts)",
    }
