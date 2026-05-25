"""
Deep Research prompt templates.

Includes prompts for topic analysis, search planning, knowledge synthesis,
and report generation.
"""


class DeepResearchPrompts:
    """Prompt templates used by the deep research pipeline."""

    # ==================== Research Planning ====================

    TOPIC_ANALYSIS = """
Analyze the following research topic and help create a research plan.

## Research Topic
{topic}

## User Background
{user_profile}

## Existing Related Knowledge (if any)
{existing_knowledge}

## Analyze and return JSON

```json
{{
    "refined_topic": "Refined wording of the topic",
    "complexity": "simple | medium | complex",
    "domain": "Domain classification (for example: technology, business, science)",
    "research_goals": [
        "Research goal 1",
        "Research goal 2"
    ],
    "key_questions": [
        "Key question 1",
        "Key question 2"
    ],
    "suggested_title": "Suggested report title"
}}
```
"""

    SUBTOPIC_GENERATION = """
Break the following research topic into searchable subtopics.

## Topic
{topic}

## Research Goals
{research_goals}

## User Interests
{user_interests}

## Requirements
1. Generate {max_subtopics} subtopics
2. Each subtopic should be independently searchable
3. Sort them by importance with priority scores from 1-10
4. Generate 2-3 search queries for each subtopic

## Return JSON

```json
{{
    "subtopics": [
        {{
            "name": "Subtopic name",
            "priority": 8,
            "queries": [
                "Search query 1",
                "Search query 2"
            ]
        }}
    ],
    "outline": {{
        "sections": [
            "Section 1: xxx",
            "Section 2: xxx"
        ]
    }}
}}
```
"""

    # ==================== Search and Extraction ====================

    SOURCE_QUALITY_ASSESSMENT = """
Assess the quality of the following search result source.

## Search Query
{query}

## Source Information
Title: {title}
URL: {url}
Snippet: {snippet}

## Content Preview
{content_preview}

## Evaluation Dimensions
1. relevance: relevance to the query topic (0-1)
2. credibility: source credibility (0-1)
3. freshness: timeliness of the information (0-1)

## Return JSON

```json
{{
    "relevance": 0.8,
    "credibility": 0.7,
    "freshness": 0.9,
    "reason": "Brief explanation"
}}
```
"""

    FACT_EXTRACTION = """
Extract topic-relevant factual information from the content below.

## Research Topic
{topic}

## Current Subtopic
{subtopic}

## Content Source
Title: {title}
URL: {url}

## Content
{content}

## Requirements
1. Extract specific, verifiable facts
2. Preserve the original meaning and do not add speculation
3. Each fact should be independently understandable
4. Assign a confidence score (0-1) to each fact
5. Try to extract the publication date from the content if available, using YYYY-MM-DD format

## Return JSON

```json
{{
    "facts": [
        {{
            "content": "Extracted fact",
            "confidence": 0.85,
            "summary": "One-sentence summary"
        }}
    ],
    "publish_date": "2024-01-15 or an empty string if unknown"
}}
```
"""

    DISCOVER_NEW_DIRECTIONS = """
Discover new research directions based on the information collected so far.

## Research Topic
{topic}

## Covered Subtopics
{covered_subtopics}

## Recently Extracted Facts
{recent_facts}

## Analysis
1. Are there newly surfaced questions worth exploring?
2. Are there important missing aspects?
3. Are there areas that deserve deeper investigation?

## Return JSON

```json
{{
    "new_questions": [
        "New question 1",
        "New question 2"
    ],
    "new_subtopics": [
        "Newly discovered subtopic"
    ],
    "suggested_queries": [
        {{
            "query": "Suggested search query",
            "purpose": "Purpose",
            "priority": 7
        }}
    ],
    "should_continue": true,
    "reason": "Reason for continuing or stopping"
}}
```
"""

    # ==================== Knowledge Synthesis ====================

    IDENTIFY_CONFLICTS = """
Identify contradictions or conflicts among the following facts.

## Research Topic
{topic}

## Collected Facts
{facts}

## Analysis Requirements
1. Identify statements that contradict each other
2. Note the conflicting sources
3. Provide possible explanations or resolution suggestions

## Return JSON

```json
{{
    "conflicts": [
        {{
            "fact1": "First conflicting statement",
            "fact2": "Second conflicting statement",
            "source1": "Source 1",
            "source2": "Source 2",
            "resolution": "Possible explanation or recommendation"
        }}
    ],
    "consistency_score": 0.85,
    "notes": "Other observations"
}}
```
"""

    CALCULATE_COVERAGE = """
Assess how well the current research covers the topic.

## Research Topic
{topic}

## Planned Subtopics
{planned_subtopics}

## Fact Counts By Subtopic
{facts_by_subtopic}

## Evaluation
1. Which subtopics are well covered?
2. Which subtopics need more support?
3. How strong is the overall coverage?

## Return JSON

```json
{{
    "coverage_by_subtopic": {{
        "Subtopic 1": 0.8,
        "Subtopic 2": 0.3
    }},
    "total_coverage": 0.65,
    "gaps": [
        "Gap 1: xxx",
        "Gap 2: xxx"
    ],
    "sufficient": false
}}
```
"""

    # ==================== Incremental Search ====================

    INCREMENTAL_GAP_ANALYSIS = """
Analyze the gap between existing research content and a new research topic.

## New Research Topic
{new_topic}

## Existing Related Research
Topic: {existing_topic}
Similarity: {similarity}

## Existing Facts ({fact_count} items)
{existing_facts}

## Target Subtopics
{target_subtopics}

## Analyze
1. Which existing facts can be reused directly?
2. What information is still missing?
3. Which supplementary search queries are needed?

## Return JSON

```json
{{
    "reusable_facts": ["fact_id_1", "fact_id_2"],
    "covered_subtopics": ["Subtopic 1", "Subtopic 2"],
    "missing_subtopics": ["Subtopic 3", "Subtopic 4"],
    "supplementary_queries": [
        {{"query": "Supplementary search query", "purpose": "Purpose", "priority": 8}}
    ],
    "estimated_coverage": 0.6
}}
```
"""

    # ==================== Report Generation ====================

    REPORT_GENERATION = """
You are a professional research report writer. Based on the following research results, generate a high-quality report for the user.

## Research Topic
{topic}

## Research Goals
{research_goals}

## Report Outline
{outline}

## User Profile
{user_profile}

## Collected Facts
{facts}

## Sources (References)
{sources}

## Identified Conflicts / Disputes
{conflicts}

## Requirements
1. Adjust the report's style, depth, and emphasis based on the user profile
   - Expand more in areas related to the user's interests
   - Adapt the wording to the user's communication style
   - Highlight points that matter for the user's role and goals
2. Base the report strictly on the provided facts; do not invent information
3. Important: cite sources in the main body with markers such as [1], [2], and ensure every key factual claim is cited
4. If there are conflicting viewpoints, present them objectively
5. Add a "References" section at the end listing all cited sources
6. Add the limitations of the research at the end

## Reference Format Example
```
## References

[1] Title. Publication date. URL
[2] Title. Accessed on YYYY-MM-DD. URL
```

Generate the report in Markdown:
"""

    GENERATE_SUMMARY = """
Generate a concise executive summary for the following research report.

## Report Content
{report_content}

## User Background
{user_profile}

## Requirements
1. 150-300 words
2. Highlight the core findings
3. Make it suitable for quickly understanding the report
4. Adjust the wording based on the user background

Generate the summary:
"""

    GENERATE_METHODOLOGY = """
Generate a methodology note for the research report.

## Research Process
- Search iterations: {iteration_count}
- Number of sources: {source_count}
- Number of extracted facts: {fact_count}
- Research duration: {duration}

## Generate a short methodology statement (within 100 words):
"""
