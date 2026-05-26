// Doctor Who IR System - Client-side JavaScript

// Method information
const methodInfo = {
    'boolean': {
        name: 'Boolean Search',
        description: 'Exact token matching using inverted index. Fast and precise.',
        hint: 'Exact keywords work best'
    },
    'bm25': {
        name: 'BM25 Search',
        description: 'Probabilistic ranking considering term frequency and document length.',
        hint: 'Natural language queries work well'
    },
    'semantic': {
        name: 'Semantic Search',
        description: 'Dense vector similarity using SentenceTransformers. Understands meaning.',
        hint: 'Conceptual and paraphrased queries work best'
    },
    'faiss': {
        name: 'FAISS Search',
        description: 'Fast approximate nearest neighbor search using HNSW index.',
        hint: 'Similar to semantic, but optimized for speed'
    },
    'fused': {
        name: 'Fused Search',
        description: 'Combines BM25 and semantic results using Reciprocal Rank Fusion.',
        hint: 'Combines strengths of both methods'
    },
    'rag': {
        name: 'RAG (AI Generated)',
        description: 'Retrieves relevant episodes and uses Ollama LLM to generate a contextual answer.',
        hint: 'Ask questions in natural language. Requires Ollama running on localhost:11434'
    }
};

// Event listeners
document.getElementById('queryInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

document.getElementById('methodSelect').addEventListener('change', updateMethodHint);

// Initialize
updateMethodHint();

/**
 * Update method hint text
 */
function updateMethodHint() {
    const method = document.getElementById('methodSelect').value;
    const hint = methodInfo[method]?.hint || '';
    const hintEl = document.getElementById('methodHint');
    hintEl.textContent = hint;
    hintEl.className = 'form-text text-muted';
}

/**
 * Perform search
 */
async function performSearch() {
    const query = document.getElementById('queryInput').value.trim();
    const method = document.getElementById('methodSelect').value;

    // Validation
    if (!query) {
        showError('Please enter a query');
        return;
    }

    if (query.length < 2) {
        showError('Query must be at least 2 characters');
        return;
    }

    // Show loading state
    showLoading();

    try {
        let response;
        if (method === 'rag') {
            response = await fetch('/api/rag', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });
        } else {
            response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, method })
            });
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Search failed');
        }

        const data = await response.json();

        if (method === 'rag') {
            displayRagResults(data);
        } else {
            displaySearchResults(data);
        }

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

/**
 * Display search results
 */
function displaySearchResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const searchResults = document.getElementById('searchResults');
    const searchResultsBody = document.getElementById('searchResultsBody');
    const resultCount = document.getElementById('resultCount');
    const resultMethod = document.getElementById('resultMethod');

    // Hide RAG results, show search results
    document.getElementById('ragResults').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';

    // Update header
    resultCount.textContent = data.count;
    resultMethod.textContent = `Method: ${methodInfo[data.method]?.name || data.method}`;

    // Build results HTML
    let resultsHTML = '';
    if (data.results.length === 0) {
        resultsHTML = '<p class="text-muted">No results found for this query.</p>';
    } else {
        data.results.forEach((result, index) => {
            resultsHTML += `
                <div class="result-item mb-3">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${index + 1}. <strong>${result}</strong></h6>
                            <small class="text-muted">Episode ID</small>
                        </div>
                        <span class="badge bg-secondary">#${index + 1}</span>
                    </div>
                </div>
            `;
        });
    }

    searchResultsBody.innerHTML = resultsHTML;
    searchResults.style.display = 'block';
    resultsSection.style.display = 'block';
}

/**
 * Display RAG results
 */
function displayRagResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const ragResults = document.getElementById('ragResults');
    const ragAnswer = document.getElementById('ragAnswer');
    const ragSourcesBody = document.getElementById('ragSourcesBody');

    // Hide search results
    document.getElementById('searchResults').style.display = 'none';

    // Check for error
    if (data.error || !data.has_answer) {
        showError(data.error || 'Failed to generate answer. Make sure Ollama is running.');
        return;
    }

    // Display answer
    ragAnswer.innerHTML = escapeHtml(data.answer);

    // Display retrieved sources
    let sourcesHTML = '';
    if (data.retrieved_docs && data.retrieved_docs.length > 0) {
        data.retrieved_docs.forEach((doc, index) => {
            sourcesHTML += `
                <div class="source-item mb-2">
                    <span class="badge bg-info">${index + 1}</span>
                    <strong>${escapeHtml(doc)}</strong>
                </div>
            `;
        });
    } else {
        sourcesHTML = '<p class="text-muted">No source documents.</p>';
    }

    ragSourcesBody.innerHTML = sourcesHTML;
    ragResults.style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
    resultsSection.style.display = 'block';
}

/**
 * Show loading state
 */
function showLoading() {
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');

    document.getElementById('searchResults').style.display = 'none';
    document.getElementById('ragResults').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';

    loadingSpinner.style.display = 'block';
    resultsSection.style.display = 'block';
}

/**
 * Show error message
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');

    errorText.textContent = message;
    errorMessage.style.display = 'block';
    loadingSpinner.style.display = 'none';

    document.getElementById('searchResults').style.display = 'none';
    document.getElementById('ragResults').style.display = 'none';

    resultsSection.style.display = 'block';
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
