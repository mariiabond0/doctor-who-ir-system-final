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

    if (!query) {
        showError('Please enter a query');
        return;
    }

    setLoading(true);

    try {
        const url = method === 'rag' ? '/api/rag' : '/api/search';

        const response = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(method === 'rag' ? { query } : { query, method })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Search failed');
        }

        const data = await response.json();

        if (method === 'rag') {
            displayRagResults(data);
        } else {
            displaySearchResults(data);
        }

    } catch (err) {
        showError(err.message || String(err));
    } finally {
        setLoading(false);
    }
}

/**
 * Display search results
 */
function displaySearchResults(data) {
    document.getElementById('loadingSpinner').style.display = 'none';

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

    // Build results HTML
  // Build results HTML
    let resultsHTML = '';
    if (data.results.length === 0) {
        resultsHTML = '<p class="text-muted">No results found for this query.</p>';
    } else {
        data.results.forEach((result, index) => {
            // Safely sanitize and escape the DB content to protect against XSS
            const title = escapeHtml(result.title || "Untitled Episode");
            const description = escapeHtml(result.description || "No description provided.");
            const season = escapeHtml(String(result.season));
            const number = escapeHtml(String(result.number));
            const docId = escapeHtml(String(result.doc_id));

            resultsHTML += `
                <div class="result-item mb-4 p-3 border rounded bg-light shadow-sm">
                    <div class="d-flex justify-content-between align-items-start">
                        <div style="max-width: 85%;">
                            <h5 class="mb-1 text-primary">
                                <strong>S${season}E${number}: ${title}</strong>
                            </h5>
                            <small class="text-muted d-block mb-2">System ID: <code>${docId}</code></small>
                            <p class="mb-0 text-dark" style="font-size: 0.95rem; line-height: 1.4;">
                                ${description}
                            </p>
                        </div>
                        <span class="badge bg-dark px-2 py-1">Rank #${index + 1}</span>
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
    document.getElementById('loadingSpinner').style.display = 'none';

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

    // Display enriched source documents
    let sourcesHTML = '';
    if (data.retrieved_docs && data.retrieved_docs.length > 0) {
        data.retrieved_docs.forEach((doc, index) => {
            // Safely sanitize and escape the metadata properties
            const title = escapeHtml(doc.title || "Untitled Episode");
            const description = escapeHtml(doc.description || "No content summary.");
            const season = escapeHtml(String(doc.season));
            const number = escapeHtml(String(doc.number));
            const docId = escapeHtml(String(doc.doc_id));

            sourcesHTML += `
                <div class="source-item mb-3 p-3 border rounded bg-light shadow-sm">
                    <div class="d-flex justify-content-between align-items-start">
                        <div style="max-width: 90%;">
                            <h6 class="mb-1 text-info">
                                <strong>[Source #${index + 1}] S${season}E${number}: ${title}</strong>
                            </h6>
                            <small class="text-muted d-block mb-1">System ID: <code>${docId}</code></small>
                            <p class="mb-0 text-secondary" style="font-size: 0.9rem; line-height: 1.4;">
                                ${description}
                            </p>
                        </div>
                    </div>
                </div>
            `;
        });
    } else {
        sourcesHTML = '<p class="text-muted">No source documents retrieved.</p>';
    }

    ragSourcesBody.innerHTML = sourcesHTML;
    ragResults.style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
    resultsSection.style.display = 'block';
}

/**
 * Show loading state
 */
function setLoading(isLoading) {
    const spinner = document.getElementById('loadingSpinner');
    const btn = document.getElementById('searchBtn');
    const resultsSection = document.getElementById('resultsSection');

    if (isLoading) {
        resultsSection.style.display = 'block';
        spinner.style.display = 'block';
        btn.disabled = true;
        btn.innerText = 'Searching...';
    } else {
        spinner.style.display = 'none';
        btn.disabled = false;
        btn.innerText = '🔍 Search';
    }
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
