# Chatbot RAG vs Conversational Differentiation

## Overview

The chatbot uses a two-stage detection system to determine whether a question requires RAG (Retrieval-Augmented Generation) search through Discord message history, or can be answered conversationally.

## Detection Flow

```
User Message
    ↓
Is it a question? (_is_question)
    ↓ Yes
Does it need RAG? (_needs_rag)
    ↓ Yes                    ↓ No
Use RAG Pipeline      Use Conversational Chat
```

## Stage 1: Question Detection (`_is_question`)

Detects if a message is a question using:
- Question marks (`?`)
- Question starters (what, when, where, who, why, how, etc.)
- Question phrases ("tell me", "explain", "do you know")
- Imperative questions ("explain X", "tell me about Y")

## Stage 2: RAG Need Detection (`_needs_rag`)

Determines if a question requires searching Discord history.

### ✅ RAG IS NEEDED For:

1. **User Mentions**
   - `@Alice what did you say?` → RAG
   - `@Bob how are you?` → RAG (even conversational, mentions trigger RAG)

2. **Temporal References**
   - `What did we decide yesterday?` → RAG
   - `What was discussed last week?` → RAG
   - `Did we talk about this earlier?` → RAG
   - Keywords: yesterday, last week, earlier, before, previously, recently

3. **History Keywords**
   - `What did Alice say?` → RAG
   - `Who mentioned the meeting?` → RAG
   - `What did we decide?` → RAG
   - Keywords: say/said, mention, discuss, decide, plan, agree, conversation

4. **People Indicators**
   - `What did they say?` → RAG
   - `Who said that?` → RAG
   - `Did anyone discuss this?` → RAG

5. **Decision/Plan Keywords**
   - `What was the decision?` → RAG
   - `What did we agree on?` → RAG
   - `What was the plan?` → RAG

### ❌ RAG IS NOT NEEDED For:

1. **Greetings & Small Talk**
   - `How are you?` → Chat
   - `What's up?` → Chat
   - `Hi there!` → Chat
   - `Hello!` → Chat

2. **General Conversational Questions**
   - `What can you do?` → Chat
   - `Who are you?` → Chat
   - `Can you help me?` → Chat

3. **General Knowledge Questions**
   - `What is Python?` → Chat
   - `How does this work?` → Chat

## Examples

### Conversational (No RAG)
```
User: "How are you?"
Bot: [Chat mode] "I'm doing great, thanks for asking! How can I help you?"

User: "What's up?"
Bot: [Chat mode] "Not much! Just here to help answer questions."

User: "Hi there!"
Bot: [Chat mode] "Hello! How can I assist you today?"
```

### RAG Required
```
User: "What did Alice say about the database?"
Bot: [RAG mode] Searches message history for Alice's messages about database
     "Based on the conversation history, Alice mentioned that..."

User: "@Bob what did you decide?"
Bot: [RAG mode] Filters to Bob's messages and searches
     "Bob decided to use PostgreSQL for the new project..."

User: "What did we discuss yesterday?"
Bot: [RAG mode] Searches for messages from yesterday
     "Yesterday, the team discussed..."
```

## Implementation Details

The `_needs_rag()` method uses pattern matching and keyword detection:

1. **Priority 1**: User mentions → Always RAG
2. **Priority 2**: Conversational patterns → Check for override keywords
3. **Priority 3**: Temporal/history keywords → RAG
4. **Priority 4**: People/decision keywords → RAG
5. **Default**: Chat mode (safer - avoids unnecessary searches)

## Benefits

1. **Cost Efficiency**: Avoids expensive RAG searches for simple greetings
2. **Better UX**: Natural responses for conversational questions
3. **Accurate RAG**: Only searches when history is actually needed
4. **Flexible**: Can be tuned by adjusting keyword lists

## Configuration

The behavior can be controlled via:
- `CHATBOT_USE_RAG`: Enable/disable RAG entirely
- Keyword lists in `_needs_rag()`: Can be customized for your use case

## Future Improvements

Potential enhancements:
1. **ML-based classification**: Train a model to classify questions
2. **NER (Named Entity Recognition)**: Better detection of people/entities
3. **Intent classification**: More sophisticated intent detection
4. **User feedback**: Learn from user corrections

