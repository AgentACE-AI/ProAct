"""
Prompt 模板。

所有 Agent 使用的 Prompt 模板集中管理。
"""


class Prompts:
    """所有 Prompt 模板"""

    # ===== 话题检测 =====
    TOPIC_MONITOR = """You are a topic monitoring assistant. Your task is to determine whether the user's new message is related to the current topic, or whether it returns to a previously discussed topic.

Current topic: {current_topic}
Conversation history for the current topic:
{conversation_history}

List of topics the user has previously discussed:
{existing_topics}

User's new message: {new_message}

Determine:
1. Whether the user's new message is still about the current topic
2. If not, whether it returns to a previously discussed topic
3. If neither, this is an entirely new topic

Output format (JSON):
{{
    "is_same_topic": true/false,
    "is_returning_topic": true/false,
    "returning_topic_name": "Name of the returning topic if applicable, otherwise null",
    "new_topic": "Short description of the new topic if applicable, otherwise null",
    "reason": "Reasoning for the determination"
}}

Return only JSON, no other text."""

    # ===== 规划 =====
    PLANNING = """You are a planning assistant. Your task is to determine whether the information in the current memory system is sufficient to answer the user's question, and decide the next course of action.

User question: {user_query}

Relevant information from the memory system:
{memory_content}

Analyze and determine:
1. Is the information in memory sufficient to answer the user's question?
2. If not, what content needs to be searched?

Output format (JSON):
{{
    "memory_sufficient": true/false,
    "reason": "Reasoning for the determination",
    "search_queries": ["search keyword 1", "search keyword 2"]
}}

Return only JSON, no other text."""

    # ===== 回复生成 =====
    ASSISTANT = """You are a personalized AI assistant. Based on the user's question and relevant memory information, provide a helpful response.

User profile:
{user_profile}

Relevant memories:
{relevant_memories}

User's question: {user_query}

Based on the above information, provide a personalized response. Requirements:
1. Tailor the response to the user's specific situation
2. Prioritize using concrete facts from memory to answer; do not ignore existing memory information
3. If memory contains specific information relevant to the question (names, numbers, dates, locations, etc.), you must reference it in your response
4. Use a friendly and natural tone
5. Only state that information is missing and provide general advice when memory truly contains no relevant information"""

    # ===== 记忆提取 =====
    MEMORY_EXTRACTION = """Analyze the following conversation and extract important information.

Topic: {topic}
Conversation:
{conversation}

Extract the following:
1. Topic summary: An overview of the main content discussed
2. User preferences: Likes, habits, opinions expressed by the user
3. Key information: Important facts, data, and suggestions mentioned in the conversation
4. Potential interests: Topics the user might be interested in next, based on the conversation
5. Predicted questions: Follow-up questions the user might ask

Output format (JSON):
{{
    "topic_summary": "Topic summary",
    "user_preferences": ["preference 1", "preference 2"],
    "key_information": ["key info 1", "key info 2"],
    "potential_interests": ["potential interest 1", "potential interest 2"],
    "predicted_questions": ["predicted question 1", "predicted question 2"]
}}

Return only JSON, no other text."""

    # ===== 用户画像提取 =====
    PROFILE_EXTRACTION = """Analyze the following conversation and extract the user's personal information and characteristics.

Conversation:
{conversation}

Extract the following user information from the conversation (if mentioned):

Output format (JSON):
{{
    "role": "User's identity/occupation (e.g., student, engineer), null if unknown",
    "age": "Inferred age or age range, null if unknown",
    "location": "User's location, null if unknown",
    "interests": ["Interests and hobbies expressed by the user"],
    "goals": ["User's goals or needs"],
    "challenges": ["Challenges or difficulties the user faces"],
    "traits": ["User's personality traits"],
    "communication_style": ["User's communication style characteristics"]
}}

Note: Only extract information that is explicitly mentioned or can be reasonably inferred from the conversation. Do not speculate. If a field cannot be determined from the conversation, set it to null or an empty list.

Return only JSON, no other text."""

    # ===== 增量画像提取 =====
    INCREMENTAL_PROFILE_EXTRACTION = """Based on the following conversation context and latest interaction, extract user profile information and update the summary.

Topic: {topic}

Historical summary:
{running_summary}

Latest interaction:
User: {user_query}
Assistant: {assistant_response}

Extract the following information:

Output format (JSON):
{{
    "profile_updates": {{
        "role": "User's identity/occupation (e.g., student, engineer), null if no new info",
        "age": "Inferred age or age range, null if no new info",
        "location": "User's location, null if no new info",
        "interests": ["Newly discovered interests from this interaction"],
        "goals": ["Newly discovered goals or needs from this interaction"],
        "challenges": ["Newly discovered challenges from this interaction"],
        "traits": ["Newly discovered personality traits from this interaction"],
        "communication_style": ["Newly discovered communication style from this interaction"]
    }},
    "updated_summary": "Updated conversation summary (integrate historical summary + latest interaction highlights, keep under 200 words)",
    "key_info": ["Key information points from this interaction"],
    "user_sentiment": {{
        "label": "User sentiment label, choose from: surprise/anger/sadness/joy/fear/neutral/disgust",
        "score": "Sentiment intensity score, -1.0 (strongly negative) to 1.0 (strongly positive), e.g., surprise is typically positive 0.5-0.8",
        "reason": "Brief explanation for the sentiment determination"
    }},
    "extracted_facts": [
        {{
            "entity": "Entity name (e.g., Mom, brother, friend Alice, Company X)",
            "entity_type": "person/place/organization/event/other",
            "attributes": {{"attribute_name": "attribute_value"}},
            "relationship": "Relationship to the user (e.g., mother, brother, friend, colleague)"
        }}
    ]
}}

## Sentiment label descriptions
- surprise: surprised, curious, unexpected, shocked
- anger: angry, frustrated, dissatisfied, irritated
- sadness: sad, disappointed, regretful, dejected
- joy: happy, satisfied, excited, pleased
- fear: afraid, worried, anxious, nervous
- neutral: neutral, calm, objective statement
- disgust: disgusted, repulsed, displeased

## Fact extraction guidelines
Extract specific factual information mentioned by the user, including:
- People: family members, friends, colleagues, along with their attributes (age, occupation, traits, etc.)
- Places: specific locations or cities mentioned by the user
- Organizations: companies, schools, institutions, etc.
- Events: important events or experiences mentioned by the user

Examples:
- "My mom is a doctor" -> entity:"Mom", entity_type:"person", attributes:{{"occupation":"doctor"}}, relationship:"mother"
- "My brother is 20 years old" -> entity:"brother", entity_type:"person", attributes:{{"age":"20"}}, relationship:"brother"
- "My friend Alice works at Google" -> entity:"friend Alice", entity_type:"person", attributes:{{"company":"Google"}}, relationship:"friend"

Notes:
1. profile_updates should only include **newly discovered** information from this interaction; do not repeat information already in the historical summary
2. updated_summary should retain key points from the historical summary while incorporating highlights from the latest interaction
3. If the historical summary is empty, generate an initial summary based on the current interaction
4. user_sentiment should be based on the emotional tone of the user's message, not the assistant's response
5. extracted_facts should only include facts explicitly mentioned by the user; do not speculate
6. If there are no new facts, extracted_facts should be an empty list
7. Return only JSON, no other text"""

    # ===== 主动服务 =====
    MEMORY_CRITIC = """You are a knowledge validation expert. Perform a comprehensive analysis of the following memory content.

Topic: {topic}
Memory content:
{memory_content}

Analyze from two dimensions:

**Dimension 1: Knowledge Gap Detection**
- Are there vague descriptions (e.g., "approximately", "possibly")?
- Is information about key entities missing?
- Are specific details (e.g., numbers, steps, version numbers) complete?

**Dimension 2: Critical Validation**
- Are viewpoints overly one-sided?
- Are there obvious counterarguments being overlooked?
- Is the supporting evidence sufficiently strong?

Output format (JSON):
{{
    "status": "PASS/HAS_GAPS/HAS_ISSUES",
    "confidence_score": 0-100,
    "knowledge_gaps": [
        {{"type": "missing_detail/vague_description/outdated/unverified", "description": "Gap description", "search_query": "Suggested search query"}}
    ],
    "logical_issues": [
        {{"type": "one_sided/missing_counter/logic_gap/weak_evidence", "description": "Issue description", "search_query": "Suggested search query"}}
    ],
    "summary": "Overall assessment"
}}

Return only JSON, no other text."""

    # ===== 研究主题预测 =====
    RESEARCH_TOPIC_PREDICTION = """Based on the user profile and recent interaction history, predict research topics the user might need.

## User Profile
{user_profile}

## Recent Interaction History
{recent_interactions}

## Recently Discussed Topics
{recent_topics}

## Research History (avoid duplicates)
{research_history}

## Prediction Requirements

Predict from two perspectives:

**1. Scenario Prediction (scenario)**
Identify what the user is "currently doing" and predict potential questions or information needs.
- Example: User mentions "recently adopted a Shiba Inu" -> Predict "Shiba Inu care guidelines"
- Example: User mentions "planning to learn Rust" -> Predict "Rust beginner learning path"

**2. Related Expansion (related)**
Identify domains the user frequently focuses on and recommend related topics.
- Example: User frequently discusses agent planning -> Recommend "agent reasoning techniques"
- Example: User focuses on React performance optimization -> Recommend "React 18 concurrent features"

## Notes
- Only predict topics with clear supporting evidence; do not guess
- Avoid duplicating research history topics
- Each prediction must have a clear trigger basis
- Prioritize information the user is likely to need in the near term

Output format (JSON):
{{
    "predictions": [
        {{
            "topic": "Predicted research topic",
            "type": "scenario/related",
            "reason": "Prediction rationale (why the user needs this information)",
            "confidence": 0.0-1.0,
            "trigger": "Key information that triggered the prediction (what the user said/did)"
        }}
    ]
}}

Notes:
- If there is insufficient evidence for predictions, return an empty list {{"predictions": []}}
- Return at most 3 predictions
- Do not return predictions with confidence below 0.6

Return only JSON, no other text."""

    # ===== 研究价值评估 =====
    RESEARCH_VALUE_EVALUATION = """Evaluate whether the following research topic is worth executing a search for.

## Candidate Research Topic
- Topic: {topic}
- Source: {topic_source}
- Prediction rationale: {topic_reason}
- Prediction confidence: {topic_confidence}

## User Profile
{user_profile}

## Existing Knowledge on This Topic
{existing_knowledge}

## Research History (avoid duplicates)
{research_history}

## Evaluation Dimensions

Evaluate the research value of this topic across four dimensions:

**1. User Relevance (user_relevance)**
- Is this topic related to the user's interests, goals, or current activities?
- Has the user shown recent explicit interest in this area?

**2. Knowledge Gap (knowledge_gap)**
- Is the existing knowledge sufficiently complete?
- Are there obvious information gaps?

**3. Incremental Value (incremental_value)**
- Can a search yield new, valuable information?
- Would results largely overlap with existing knowledge?
- Is this topic too similar to previous research?

**4. Timeliness (timeliness)**
- Does this information have time-sensitive requirements?
- Is now an appropriate time to search?

## Decision Criteria
- Only recommend research when the user genuinely might need this information
- If existing knowledge is sufficient or incremental value is low, do not recommend research
- Prefer missing an opportunity over producing low-quality proactive searches

Output format (JSON):
{{
    "should_research": true/false,
    "score": 0-100,
    "reason": "Comprehensive evaluation rationale",
    "factors": {{
        "user_relevance": 0-100,
        "knowledge_gap": 0-100,
        "incremental_value": 0-100,
        "timeliness": 0-100
    }}
}}

Score reference:
- 80-100: Strongly recommend research; user clearly needs this
- 60-79: Recommend research; has reasonable value
- 40-59: Optional research; moderate value
- 0-39: Do not recommend research; insufficient value

Return only JSON, no other text."""

    PUSH_SCORE = """Evaluate the push value and interruption cost of the following report.

## Report Information
- Topic: {report_topic}
- Summary: {report_summary}

## User Information
- Interest areas: {user_interests}
- Current conversation: {current_context}

## User Activity Metrics
- Time since last interaction: {seconds_since_last_interaction} seconds
- Interactions in the last 5 minutes: {recent_interaction_count}

## Evaluate the following separately

**Value (Information value 0-100)**
Factors to consider:
- Relevance to the user's interests
- Timeliness and urgency of the information
- Potential benefit to the user

**Cost (Interruption cost 0-100)**
Factors to consider:
- User activity level (more frequent interactions and shorter time since last interaction = higher interruption cost)
- Whether the report topic is related to the current conversation
- Whether the interruption would cause annoyance

## Output format (JSON)
{{
    "value": 0-100,
    "cost": 0-100,
    "reason": "Brief scoring rationale"
}}

Return only JSON, no other text."""

    # ===== 搜索 =====
    SEARCH_PLANNING = """Plan a search strategy for the specified topic.

Topic: {topic}
Search purpose: {purpose}
User background:
{user_profile}
Existing information:
{existing_knowledge}

Develop a search plan:
1. Analyze the complexity of the search task (simple/comprehensive)
2. Generate a list of search queries
3. Determine search priorities

Output format (JSON):
{{
    "mode": "simple/comprehensive",
    "queries": [
        {{"query": "search term", "purpose": "purpose", "priority": 1-10}}
    ],
    "overview": "Search strategy description"
}}

Return only JSON, no other text."""

    # ===== 报告 =====
    REPORT_GENERATION = """Based on the following search results, generate a research report.

Topic: {topic}

Search results and web content:
{search_results}

User background:
{user_profile}

Generate a research report with the following requirements:
1. Comprehensive content covering the main information found
2. Clear structure using markdown format
3. Personalized to the user's situation if background information is available
4. Include key findings and practical recommendations
5. Cite information sources

Suggested report structure:
- Overview
- Key Findings
- Detailed Content
- Recommendations / Conclusions
- References

Output the report content directly."""

    # ===== 知识处理 =====
    DUPLICATE_DECISION = """Determine whether the following two pieces of content are duplicates and how they should be handled.

Existing content:
{existing_content}

New content:
{new_content}

Determine:
1. Whether the two pieces discuss the same topic or fact
2. If they are duplicates, how the new content should be handled

Output format (JSON):
{{
    "is_duplicate": true/false,
    "decision": "skip/replace/merge",
    "reason": "Reason for the decision"
}}

Decision meanings:
- skip: The new content adds no extra value and should be skipped
- replace: The new content is more complete or newer and should replace the old content
- merge: Both are valuable and should be merged

Return only JSON, no other text."""

    KNOWLEDGE_MERGE = """Merge the following two pieces of content into a more complete version.

Content 1:
{content1}

Content 2:
{content2}

Requirements:
1. Keep the key information from both pieces
2. Remove duplicated content
3. Preserve logical coherence
4. Keep the writing concise

Output only the merged content, with no prefix or explanation."""

    # ===== 话题摘要更新 =====
    TOPIC_SUMMARY_UPDATE = """Update the summary for a topic that has evolved across multiple turns.

Topic name: {topic}

Existing summary:
{old_summary}

New conversation content:
{new_conversation}

Generate an updated topic summary. Requirements:
1. Integrate both the previous summary and the new conversation content
2. Keep it concise and focused on the main points
3. Reflect how the discussion has evolved

Output only the updated summary, with no prefix or explanation."""

    # ===== 内容摘要 =====
    CONTENT_SUMMARY = """Generate a concise summary of the following webpage content for later retrieval.

Webpage URL: {url}
Webpage content:
{content}

Requirements:
1. Summarize the main content and key information on the page
2. Keep the summary within roughly 200-500 words
3. Preserve key facts, data points, and viewpoints
4. Make it easy to retrieve later using keywords

Output only the summary content, with no prefix or explanation."""

    # ===== 过期信息检测 =====
    STALE_INFO_CHECK = """Check whether the following memory content may be outdated.

Topic: {topic}
Memory creation time: {created_at}
Memory content:
{memory_content}

Current time: {current_time}

Analyze:
1. Whether this information may already be outdated
2. Which parts may need updating
3. How to verify the freshness of the information

Output format (JSON):
{{
    "is_potentially_stale": true/false,
    "stale_items": [
        {{"content": "Potentially stale content", "reason": "Reasoning", "verification_query": "Verification search query"}}
    ],
    "freshness_score": 0-100,
    "recommendation": "Update recommendation"
}}

Return only JSON, no other text."""

    # ===== 对话内需求预测 =====
    CONVERSATION_NEED_PREDICTION = """Analyze the current conversation context and predict what information the user might need next.

Current user message: {current_message}

Conversation history:
{conversation_history}

User profile:
{user_profile}

Already retrieved context:
{memory_context}

Predict what information the user is likely to need in the next 1-2 turns. Only predict needs with high confidence.

Criteria for a GOOD prediction:
1. A natural extension of the user's current question or action
2. Something the user's profile/background implies they would need
3. A commonly associated follow-up in this domain

Do NOT predict:
1. Needs unrelated to the current conversation topic
2. Information already discussed in the conversation
3. Overly generic or speculative needs

Output format (JSON):
{{
    "predicted_needs": [
        {{
            "need": "Brief description of the predicted need",
            "reason": "Why the user would likely need this information next",
            "confidence": 0.0-1.0,
            "lookup_query": "Search query to retrieve this information from memory"
        }}
    ]
}}

Return only JSON, no other text. Predict at most 3 needs."""

    # ===== 主动对话检查 =====
    INITIATIVE_CHECK = """Determine whether the system should proactively start a conversation with the user.

## Current State
- Current topic: {current_topic}
- Recent messages:
{recent_messages}

## User Information
- Interest areas: {user_interests}
- Pending information to push: {pending_info}
- Idle time: {idle_minutes} minutes

## Decision Criteria
Consider the following factors:
1. Whether the user is in a good state to receive a proactive message
2. Whether the pending information is relevant to the user's interests
3. Whether there is worthwhile information to share
4. Whether sending it would be disruptive

Output format (JSON):
{{
    "should_initiate": true/false,
    "type": "suggestion/reminder/question/info",
    "priority": "high/medium/low",
    "content": "Message content to send",
    "options": ["Option 1", "Option 2"],
    "reason": "Reason for the decision"
}}

Return only JSON, no other text."""

    # ===== 用户模拟器 =====
    USER_SIMULATOR = """You are a user simulator. Based on the following user profile, simulate this user talking with an AI assistant.

User profile:
{user_profile}

Conversation history:
{conversation_history}

Generate the next message as this user. Requirements:
1. Match the user's personality traits and speaking style
2. Stay grounded in the user's interests and current concerns
3. The user may ask a question, share a thought, or seek advice
4. The language should feel natural, like a real user

Output only the user's message, with no prefix or explanation."""

    # ===== 模拟用户画像生成 =====
    SIMULATOR_PROFILE_GENERATION = """Generate a complete simulated user profile based on the following short description.

## User Description
{description}

## Requirements
Generate a realistic, concrete, detailed profile including:
1. Background: role, experience, domain
2. Current activities: projects or learning efforts, with specific tech stack
3. Pending problems: 1-3 concrete problems, including context and blockers
4. Interests: 2-4 topics the user cares about
5. Behavior traits: communication style and information-reveal tendency

## Output Format (JSON)
{{
    "id": "Auto-generated ID (8 alphanumeric characters)",
    "name": "Display name",
    "identity": {{
        "role": "Specific role",
        "experience": "Years of experience",
        "domain": "Domain",
        "background": "One-sentence background"
    }},
    "ongoing_activities": [
        {{
            "type": "work_project/learning/side_project",
            "description": "Activity description",
            "tech_stack": ["Tech 1", "Tech 2"],
            "current_phase": "early/mid/finishing",
            "challenges": ["Challenge 1", "Challenge 2"]
        }}
    ],
    "pending_problems": [
        {{
            "description": "Problem description",
            "urgency": "high/medium/low",
            "context": "Problem context",
            "attempted": ["Attempt already made"],
            "blocker": "Current blocker"
        }}
    ],
    "interests": [
        {{
            "topic": "Topic of interest",
            "reason": "Why the user cares about it",
            "depth": "overview/deep_dive"
        }}
    ],
    "communication_style": "concise/balanced/detailed",
    "info_reveal_tendency": "explicit/implicit/minimal"
}}

Return only JSON, no other text."""

    # ===== 模拟用户 Query 生成 =====
    SIMULATOR_QUERY_GENERATION = """You are a user simulator and need to generate a realistic, natural user message.

## User You Are Playing
{user_profile}

## Conversation History
{conversation_history}

## Information Already Revealed (avoid repeating it)
{revealed_info}

## Trigger For This Turn
Type: {trigger_type}
Description: {trigger_description}

## Information-Reveal Strategy

1. **Blend information in naturally**
   - Information should appear as a natural part of the conversation
   - Good: "I ran into a performance issue while building a data pipeline."
   - Bad: "I am a backend engineer, interested in distributed systems, and now I have a question..."

2. **Reveal information based on the trigger**
   - Problem-driven: reveal background, attempted solutions, and blockers
   - Curiosity-driven: reveal why the user is interested
   - Follow-up: reveal prior progress

3. **Reveal information gradually**
   - On the first question, reveal only what is necessary
   - Add more detail only if the assistant asks follow-up questions

---

Generate one natural user message.

Output format (JSON):
{{
    "query": "The user's message",
    "revealed_info": ["Information revealed in this turn", "Another item"]
}}

Return only JSON."""

    # ===== 推送反应决策 =====
    SIMULATOR_PUSH_REACTION = """Determine which option the user would choose in response to a push notification.

## Push Content
{push_content}

## Available Options
{options}

## User's Current Focus
{user_focus}

## User Identity
{user_identity}

---

Decide which option the user is most likely to choose.

Output format (JSON):
{{
    "selected_option": "Chosen option (must be one of the provided options)",
    "reason": "Reason for the choice",
    "relevance_score": 0-100
}}

Return only JSON."""
