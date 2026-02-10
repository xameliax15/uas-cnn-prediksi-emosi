// Simplified JavaScript for Prediction-Only Interface

const API_URL = 'https://xameliax15.pythonanywhere.com';

// Emotion to Emoji mapping
const EMOTION_EMOJI = {
    'senang': '😊',
    'sedih': '😢',
    'marah': '😠',
    'takut': '😨',
    'jijik': '🤢',
    'terkejut': '😲',
    'netral': '😐'
};

// Emotion colors
const EMOTION_COLORS = {
    'senang': '#10b981',
    'sedih': '#3b82f6',
    'marah': '#ef4444',
    'takut': '#8b5cf6',
    'jijik': '#84cc16',
    'terkejut': '#f59e0b',
    'netral': '#6b7280'
};

// DOM Elements
const inputText = document.getElementById('inputText');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const errorAlert = document.getElementById('errorAlert');
const mainEmotion = document.getElementById('mainEmotion');
const confidencePercent = document.getElementById('confidencePercent');
const emojiDisplay = document.getElementById('emojiDisplay');
const probabilityBars = document.getElementById('probabilityBars');
const preprocessedText = document.getElementById('preprocessedText');

// Event Listeners
analyzeBtn.addEventListener('click', analyzeText);
clearBtn.addEventListener('click', clearInput);

// Enter key to analyze
inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeText();
    }
});

// Set example text
function setExample(text) {
    inputText.value = text;
    inputText.focus();
}

// Clear input
function clearInput() {
    inputText.value = '';
    resultSection.style.display = 'none';
    errorAlert.style.display = 'none';
    inputText.focus();
}

// Show error
function showError(message) {
    errorAlert.textContent = '❌ ' + message;
    errorAlert.style.display = 'block';
    setTimeout(() => {
        errorAlert.style.display = 'none';
    }, 5000);
}

// Analyze text
async function analyzeText() {
    const text = inputText.value.trim();

    // Validation
    if (!text) {
        showError('Silakan masukkan teks terlebih dahulu');
        return;
    }

    if (text.length < 3) {
        showError('Teks terlalu pendek. Minimal 3 karakter');
        return;
    }

    // Show loading
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    errorAlert.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message || 'Gagal menganalisis teks');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Gagal terhubung ke server. Pastikan backend sedang berjalan.');
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    const emotion = data.prediction;
    const confidence = data.confidence;
    const probabilities = data.probabilities;
    const preprocessed = data.preprocessed_text;

    // Main emotion
    mainEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    mainEmotion.style.color = EMOTION_COLORS[emotion] || '#667eea';

    // Confidence
    confidencePercent.textContent = (confidence * 100).toFixed(1) + '%';

    // Set confidence badge color
    const confidenceBadge = document.querySelector('.confidence-badge');
    if (confidence >= 0.8) {
        confidenceBadge.style.background = '#10b981'; // High confidence - green
    } else if (confidence >= 0.6) {
        confidenceBadge.style.background = '#f59e0b'; // Medium confidence - orange
    } else {
        confidenceBadge.style.background = '#ef4444'; // Low confidence - red
    }

    // Emoji
    emojiDisplay.textContent = EMOTION_EMOJI[emotion] || '🤔';

    // Probability bars
    probabilityBars.innerHTML = '';

    // Sort by probability (descending)
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([label, prob]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';

        const percentage = (prob * 100).toFixed(1);
        const color = EMOTION_COLORS[label] || '#667eea';

        probItem.innerHTML = `
            <div class="prob-header">
                <span class="prob-label">${label.charAt(0).toUpperCase() + label.slice(1)}</span>
                <span class="prob-value">${percentage}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${percentage}%; background: ${color};"></div>
            </div>
        `;

        probabilityBars.appendChild(probItem);
    });

    // Preprocessed text
    preprocessedText.textContent = preprocessed;

    // Show results with animation
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Auto-focus on load
window.addEventListener('load', () => {
    inputText.focus();
});

// Check server status on load
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_URL}/`);
        const data = await response.json();
        console.log('✅ Server status:', data);
    } catch (error) {
        console.warn('⚠️ Backend server tidak terhubung. Pastikan server berjalan di http://localhost:5000');
        showError('Backend server belum berjalan. Jalankan: python backend/main.py');
    }
}

checkServerStatus();
