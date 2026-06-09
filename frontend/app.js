// API Server Configuration
const API_BASE_URL = 'http://localhost:8000';

// App State
let activeMode = 'hybrid_rerank';

// DOM Elements
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const modeTabs = document.querySelectorAll('.mode-tab');

// Metrics DOM Elements
const retrievalTimeEl = document.getElementById('retrieval-time');
const generationTimeEl = document.getElementById('generation-time');
const promptTokensEl = document.getElementById('prompt-tokens');
const completionTokensEl = document.getElementById('completion-tokens');
const totalTokensEl = document.getElementById('total-tokens');
const promptBar = document.getElementById('prompt-bar');
const completionBar = document.getElementById('completion-bar');
const sourcesList = document.getElementById('sources-list');

// Config DOM Elements
const cfgTier = document.getElementById('cfg-tier');
const cfgCollection = document.getElementById('cfg-collection');
const cfgModel = document.getElementById('cfg-model');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    
    // Check health periodically every 15 seconds
    setInterval(checkHealth, 15000);
});

// Setup Listeners
function setupEventListeners() {
    // Mode selector tabs
    modeTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            modeTabs.forEach(t => t.classList.remove('active'));
            const selectedTab = e.currentTarget;
            selectedTab.classList.add('active');
            activeMode = selectedTab.getAttribute('data-mode');
        });
    });

    // Chat submit form
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitQuery();
    });

    // Textarea auto-resize and Enter key send
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });
    
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = queryInput.scrollHeight + 'px';
    });
}

// Helper to determine if a model has reasoning capabilities
function isReasoningModel(modelName) {
    if (!modelName) return false;
    const name = modelName.toLowerCase();
    return name.includes('deepseek') || 
           name.includes('gemma-4-26b') || 
           name.includes('o1') || 
           name.includes('o3') || 
           name.includes('reasoning') || 
           name.includes('thinking');
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (response.ok && data.status === 'healthy') {
            updateStatus('healthy', 'Healthy (Connected to Qdrant)');
        } else if (response.ok && data.status.includes('degraded')) {
            updateStatus('loading', 'Degraded (Collection Not Found)');
        } else {
            updateStatus('unhealthy', 'Unhealthy (Database Connected Failed)');
        }

        // Update active configuration panel
        cfgTier.textContent = data.active_tier || 'unknown';
        cfgCollection.textContent = data.collection_name || 'unknown';
        cfgTier.className = `badge tier-${data.active_tier}`;
        
        if (cfgModel) {
            cfgModel.textContent = data.llm_model ? data.llm_model.split('/').pop() : 'unknown';
        }

        // Show/hide thinking toggle depending on model capabilities
        const toggle = document.getElementById('thinking-toggle');
        if (toggle) {
            const container = toggle.closest('.selector-container');
            if (container) {
                if (isReasoningModel(data.llm_model)) {
                    container.style.display = 'block';
                } else {
                    container.style.display = 'none';
                }
            }
        }
    } catch (error) {
        updateStatus('unhealthy', 'Unhealthy (Backend Server Offline)');
        cfgTier.textContent = '--';
        cfgCollection.textContent = '--';
        if (cfgModel) cfgModel.textContent = '--';
    }
}

function updateStatus(type, message) {
    statusBadge.className = `status-badge status-${type}`;
    statusText.textContent = message;
}

// Suggestion helper
function useQuery(text) {
    queryInput.value = text;
    queryInput.dispatchEvent(new Event('input'));
    queryInput.focus();
}

// Render Markdown helper
function renderMarkdown(text) {
    if (!text) return '';
    
    // Escape HTML first to prevent XSS
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code Blocks: ```lang ... ```
    html = html.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
        const language = lang || 'python';
        return `<pre><code class="language-${language}">${code}</code></pre>`;
    });

    // Inline Code: `code`
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold: **text**
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Lists items
    html = html.replace(/^\*\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- \s*(.+)$/gm, '<li>$1</li>');
    
    // Group lists
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, (match) => {
        return `<ul>${match}</ul>`;
    });
    // Deduplicate nested lists
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Line breaks (exclude inside <pre>)
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/(<pre>[\s\S]*?<\/pre>)/g, (match) => {
        return match.replace(/<br>/g, '\n');
    });

    return html;
}

// Submit RAG query
async function submitQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Get the status of the thinking toggle
    const thinkingToggle = document.getElementById('thinking-toggle');
    const enableThinking = thinkingToggle ? thinkingToggle.checked : true;

    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Add User bubble
    appendMessage('user', query);

    // Disable input/button while loading
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Add Assistant Typing Bubble
    const typingId = appendTypingIndicator();

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: query,
                mode: activeMode,
                thinking: enableThinking
            })
        });

        // Remove Typing indicator
        document.getElementById(typingId).remove();

        if (response.ok) {
            const data = await response.json();
            
            // Add Assistant Answer bubble
            appendMessage('assistant', data.response, data.reasoning);
            
            // Update Dashboard Metrics
            updateMetrics(data.metadata, data.retrieved_documents);
        } else {
            const err = await response.json();
            appendMessage('assistant', `⚠️ Error from API: ${err.detail || response.statusText}`);
        }
    } catch (error) {
        // Remove Typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        
        appendMessage('assistant', `❌ Failed to communicate with RAG service: ${error.message}`);
    } finally {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

// Append bubble helper
function appendMessage(sender, text, reasoning = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    if (sender === 'user') {
        content.textContent = text;
    } else {
        let messageHtml = '';
        if (reasoning) {
            messageHtml += `
                <details class="reasoning-details" open>
                    <summary class="reasoning-summary">
                        <i class="fa-solid fa-brain"></i>
                        <span>Thought Process</span>
                    </summary>
                    <div class="reasoning-content-inner">${renderMarkdown(reasoning)}</div>
                </details>
            `;
        }
        messageHtml += `<div class="answer-body">${renderMarkdown(text)}</div>`;
        content.innerHTML = messageHtml;
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatWindow.appendChild(messageDiv);
    
    // Scroll to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Append Typing indicator helper
function appendTypingIndicator() {
    const typingId = `typing-${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = typingId;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatWindow.appendChild(messageDiv);
    
    // Scroll to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return typingId;
}

// Update Metrics Dashboard
function updateMetrics(metadata, documents) {
    // 1. Latency Times
    retrievalTimeEl.textContent = `${metadata.retrieval_latency_sec.toFixed(3)}s`;
    generationTimeEl.textContent = `${metadata.generation_latency_sec.toFixed(3)}s`;
    
    // 2. Tokens Count
    const usage = metadata.token_usage || {};
    const prompt = usage.prompt_tokens || 0;
    const completion = usage.completion_tokens || 0;
    const total = usage.total_tokens || (prompt + completion);
    
    promptTokensEl.textContent = prompt || '--';
    completionTokensEl.textContent = completion || '--';
    totalTokensEl.textContent = total || '--';
    
    // Update progress bars (base on max capacity scale of ~8000 tokens)
    const scaleMax = 8000;
    const promptPct = Math.min((prompt / scaleMax) * 100, 100);
    const completionPct = Math.min((completion / scaleMax) * 100, 100);
    
    promptBar.style.width = `${promptPct}%`;
    completionBar.style.width = `${completionPct}%`;
    
    // 3. Retrieved Documents
    sourcesList.innerHTML = '';
    
    if (!documents || documents.length === 0) {
        sourcesList.innerHTML = '<li class="no-sources">No documents retrieved.</li>';
        return;
    }
    
    documents.forEach(doc => {
        const li = document.createElement('li');
        li.className = 'source-item';
        
        li.innerHTML = `
            <div class="source-heading">${doc.heading}</div>
            <div class="source-page">Page: ${doc.page_id}</div>
            <a href="${doc.url}" target="_blank" class="source-link">
                <i class="fa-solid fa-arrow-up-right-from-square"></i> Open Docs
            </a>
        `;
        
        sourcesList.appendChild(li);
    });
}
