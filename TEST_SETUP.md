# Test Setup Instructions

## Current Status

✅ **Test files are syntactically correct**
- `tests/test_advanced_rag.py` - No syntax errors
- All imports are valid

❌ **pytest is not installed in the current environment**

## Setup Instructions

### Option 1: Use Virtual Environment (Recommended)

```bash
# Create virtual environment
cd /Users/youmyeongkim/projects/deep-bot
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Option 2: Install pytest with --break-system-packages (Not Recommended)

```bash
pip3 install --break-system-packages pytest pytest-asyncio pytest-mock
pytest tests/ -v
```

### Option 3: Use pipx (If Available)

```bash
brew install pipx
pipx install pytest
pytest tests/ -v
```

## Running Tests

Once pytest is installed, you can run:

```bash
# Run all tests
pytest tests/ -v

# Run only new advanced RAG tests
pytest tests/test_advanced_rag.py -v

# Run with coverage
pytest tests/ --cov=rag --cov-report=html

# Run specific test class
pytest tests/test_advanced_rag.py::TestHybridSearch -v

# Run specific test
pytest tests/test_advanced_rag.py::TestHybridSearch::test_hybrid_search_basic -v
```

## Expected Test Results

The new test suite includes:
- **22 tests** in `test_advanced_rag.py`
- Tests for hybrid search, reranking, multi-query, and HyDE
- Integration tests for pipeline

All tests use mocks, so they should run quickly without requiring:
- Actual vector database
- Real AI API calls
- Cross-encoder model downloads

## Troubleshooting

If tests fail:

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing sentence-transformers**: Required for reranking tests
   ```bash
   pip install sentence-transformers
   ```

3. **Missing rank-bm25**: Required for hybrid search
   ```bash
   pip install rank-bm25
   ```

4. **Async test errors**: Make sure `pytest-asyncio` is installed

