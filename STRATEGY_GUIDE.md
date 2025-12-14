# Complete Strategy Guide

## All Available Strategies

### Comprehensive

**`all`** - Apply ALL 14 strategies (default)
- No auto-detection, applies every strategy
- Strategies: entity_sheet, who_what_why, keyword, sentence, full_qa, relationship_mapping, timeline, conflict_plot, rule_mechanic, project_planning, requirements, task_breakdown, document_section, code_snippet
- Best for maximum retrieval coverage
- Some strategies may produce no results if content doesn't match (expected behavior)

### Basic Content Strategies

**`entity_sheet`** - Extract named entities
- Characters, locations, items, factions
- Best for: Fictional content, world-building
- Creates: Entity name → full description mappings

**`who_what_why`** - Structured factual Q&A
- Extracts: who, what, why, when, where, how
- Best for: Technical documentation, news, factual content
- Creates: Structured searchable metadata

**`keyword`** - Keyword extraction
- Identifies key terms and phrases
- Best for: Reference material, glossaries, quick lookup
- Creates: Keyword-tagged chunks

**`sentence`** - Sentence-level granularity
- Splits into individual sentences
- Best for: Precise fact retrieval, detailed search
- Creates: Fine-grained searchable units

**`full_qa`** - Generate Q&A pairs
- LLM generates natural questions and answers
- Best for: FAQ creation, chatbot training
- Creates: Canonical question-answer pairs

**`document_section`** - Markdown sections
- Splits by headings (# ## ###)
- Best for: Documentation, structured content
- Default for markdown files

**`code_snippet`** - Code extraction
- Extracts functions with summaries
- Best for: Code repositories, technical docs
- Default for code files

### Advanced Narrative Strategies

**`relationship_mapping`** - Entity relationships
- Extracts: "Entity A [relationship] Entity B because [context]"
- Best for: Understanding character dynamics, faction politics
- Example: "Character A controls Location B because of [historical reason]"

**`timeline`** - Chronological events
- Extracts events in temporal order
- Best for: Story progression, historical events
- Enables: "What happened before X?", "Events in Year Y"

**`conflict_plot`** - Narrative conflicts
- Extracts: problem, stakes, parties, status, outcome
- Best for: Plot understanding, story analysis
- Example: "Dark forces threaten the realm. Stakes: total destruction..."

**`rule_mechanic`** - World systems
- Extracts: game rules, magic systems, physics
- Best for: Game design docs, world-building
- Format: "When [trigger], then [effect]. Exceptions: [...]"

### Project Planning Strategies

**`project_planning`** - High-level planning
- Extracts: goals, scope, out-of-scope, stakeholders, constraints, risks
- Best for: Project charters, planning documents
- Creates separate chunks for goals, scope, and risks

**`requirements`** - Requirement extraction
- Categorizes: functional, non-functional, business, technical
- Includes: priority, acceptance criteria
- Best for: PRDs, specs, requirement docs

**`task_breakdown`** - Work decomposition
- Extracts: tasks, dependencies, effort, assignments
- Best for: Sprint planning, project management
- Enables: "What tasks are in frontend?", "What's assigned to X?"

## Recommended Usage Patterns

### For Fictional Content (Stories, World-Building, Game Lore)

```bash
# Comprehensive (recommended)
./ollamatui import ./lore --strategy all --verbose

# Or layer-by-layer
./ollamatui import ./lore --strategy entity_sheet
./ollamatui import ./lore --strategy relationship_mapping --force
./ollamatui import ./lore --strategy timeline --force
./ollamatui import ./lore --strategy conflict_plot --force
./ollamatui import ./lore --strategy rule_mechanic --force
```

**Result**:
- Entity lookups: "Who is Character X?"
- Relationships: "How is Character A related to Character B?"
- Timeline: "What happened before the main event?"
- Plot: "What conflicts exist?"
- Rules: "How does the magic system work?"

### For Technical Documentation

```bash
# Comprehensive
./ollamatui import ./docs --strategy all

# Or specific
./ollamatui import ./api-docs --strategy who_what_why
./ollamatui import ./api-docs --strategy code_snippet --force
```

**Result**:
- Factual Q&A: "What does this API do?"
- Code examples: "Show me authentication code"

### For Project Planning

```bash
# Full project analysis
./ollamatui import ./project-charter.md --strategy all

# Specific aspects
./ollamatui import ./prd.md --strategy requirements
./ollamatui import ./planning.md --strategy task_breakdown
```

**Result**:
- "What are the must-have requirements?"
- "What tasks depend on authentication?"
- "What are the project risks?"

### Multi-Strategy Workflow (Best Practice)

```bash
# Pass 1: Quick indexing
./ollamatui import ./knowledge-base --strategy keyword

# Pass 2: Deep extraction (force re-import)
./ollamatui import ./knowledge-base --strategy entity_sheet --force
./ollamatui import ./knowledge-base --strategy relationship_mapping --force
./ollamatui import ./knowledge-base --strategy timeline --force

# Pass 3: Comprehensive coverage
./ollamatui import ./knowledge-base --strategy all --force
```

## Strategy Selection Guide

### Choose Based on Content Type

| Content Type | Recommended Strategy | Why |
|--------------|---------------------|-----|
| Story/Lore | `all` or `relationship_mapping` + `timeline` | Captures narrative structure |
| Game Design Doc | `rule_mechanic` + `entity_sheet` | Systems and entities |
| Technical Docs | `who_what_why` + `code_snippet` | Factual + examples |
| API Documentation | `code_snippet` + `full_qa` | Code + natural Q&A |
| Project Charter | `project_planning` | Goals, scope, risks |
| Requirements Doc | `requirements` + `task_breakdown` | Specs + tasks |
| Reference/Glossary | `keyword` + `entity_sheet` | Quick lookup |
| Code Repository | `code_snippet` | Default for code |

### Choose Based on Query Pattern

| Query Pattern | Strategy | Example |
|--------------|----------|---------|
| "Who is X?" | `entity_sheet` | "Who is the Entity of Y?" |
| "What is the relationship between X and Y?" | `relationship_mapping` | "How is Alice related to Bob?" |
| "What happened in Year X?" | `timeline` | "What events occurred in 2024?" |
| "What conflicts exist?" | `conflict_plot` | "What is the main conflict?" |
| "How does X work?" | `rule_mechanic` | "How does magic work?" |
| "What are the requirements?" | `requirements` | "What are the must-have features?" |
| "What tasks are needed?" | `task_breakdown` | "What needs to be done?" |
| General questions | `all` or `full_qa` | Maximum coverage |

## Testing Strategies

```bash
# Test on a single file first
./ollamatui import ./test-file.md --strategy entity_sheet --verbose

# Verify chunks created
# In TUI, search for entities to confirm they're retrievable

# If good, apply to directory
./ollamatui import ./docs --strategy entity_sheet
```

## Performance Considerations

- **`all`** strategy: Slowest (applies all 14 strategies), but best retrieval (default)
- **Specific strategies**: Faster, targeted approach
- **Multiple passes**: Flexible, can monitor progress per strategy
- **LLM calls**: Each strategy makes 1-N LLM calls per document

Recommendation: Use `all` by default, specific strategies for targeted needs or faster imports.

## Auto-completion

```bash
# See all strategies
./ollamatui import ./file.md --strategy <TAB>

# Filter strategies
./ollamatui import ./file.md --strategy rel<TAB>
# Shows: relationship_mapping, requirements
```

## Strategy Combinations for Maximum Value

**For comprehensive knowledge base:**
```bash
./ollamatui import ./kb --strategy all
```
Applies all 14 strategies. Some may produce no results depending on content (expected).

**For specific use cases:**
- Chatbot: `full_qa` + `keyword`
- Search engine: `keyword` + `sentence`
- Story assistant: `entity_sheet` + `relationship_mapping` + `timeline`
- Project tracker: `requirements` + `task_breakdown`
- Game master tool: `rule_mechanic` + `entity_sheet` + `conflict_plot`

## Next Steps

After importing, queries in the TUI will leverage:
- Hybrid search (semantic + keyword matching)
- Metadata fields for boosting
- Canonical questions for direct matching
- Entity references for relationship queries
