/**
 * Emotion Classification Web Application
 * Frontend JavaScript
 */

const API_URL = 'https://xameliax15.pythonanywhere.com';

let lossChart = null;
let accuracyChart = null;

// Dataset Upload
async function uploadDataset() {
    const fileInput = document.getElementById('datasetFile');
    const textColumn = document.getElementById('textColumn').value;
    const labelColumn = document.getElementById('labelColumn').value;

    if (!fileInput.files.length) {
        showAlert('uploadAlert', 'Silakan pilih file dataset terlebih dahulu', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('text_column', textColumn);
    formData.append('label_column', labelColumn);

    showAlert('uploadAlert', 'Mengupload dataset...', 'info');

    try {
        const response = await fetch(`${API_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showAlert('uploadAlert', data.message, 'success');
            displayStatistics(data.statistics);
        } else {
            showAlert('uploadAlert', data.message, 'error');
        }
    } catch (error) {
        showAlert('uploadAlert', `Error: ${error.message}`, 'error');
    }
}

// Display Statistics
function displayStatistics(stats) {
    const statsGrid = document.getElementById('statsGrid');
    const labelDistributionBars = document.getElementById('labelDistributionBars');

    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Total Sampel</div>
            <div class="stat-value">${stats.total_samples}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Rata-rata Panjang Teks</div>
            <div class="stat-value">${Math.round(stats.avg_text_length)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Min Panjang</div>
            <div class="stat-value">${stats.min_text_length}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Max Panjang</div>
            <div class="stat-value">${stats.max_text_length}</div>
        </div>
    `;

    // Label distribution
    const distribution = stats.emotion_distribution;
    const maxCount = Math.max(...Object.values(distribution));

    let distributionHTML = '';
    for (const [label, count] of Object.entries(distribution)) {
        const percentage = (count / maxCount) * 100;
        distributionHTML += `
            <div class="confidence-bar">
                <div class="confidence-label">${label}</div>
                <div class="confidence-fill">
                    <div class="confidence-value" style="width: ${percentage}%">
                        ${count}
                    </div>
                </div>
            </div>
        `;
    }

    labelDistributionBars.innerHTML = distributionHTML;
    document.getElementById('statisticsSection').classList.remove('hidden');
}

// Preprocess Data
async function preprocessData() {
    const maxFeatures = parseInt(document.getElementById('maxFeatures').value);
    const useStemming = document.getElementById('useStemming').checked;
    const useStopwords = document.getElementById('useStopwords').checked;

    showAlert('preprocessAlert', 'Memproses preprocessing...', 'info');

    try {
        const response = await fetch(`${API_URL}/api/preprocess`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                max_features: maxFeatures,
                use_stemming: useStemming,
                use_stopword_removal: useStopwords
            })
        });

        const data = await response.json();

        if (data.success) {
            showAlert('preprocessAlert', data.message, 'success');
            displayPreprocessPreview(data.preview);
        } else {
            showAlert('preprocessAlert', data.message, 'error');
        }
    } catch (error) {
        showAlert('preprocessAlert', `Error: ${error.message}`, 'error');
    }
}

// Display Preprocess Preview
function displayPreprocessPreview(preview) {
    const previewContent = document.getElementById('previewContent');

    let html = '';
    preview.forEach((item, index) => {
        html += `
            <div class="result-card" style="margin-bottom: 1rem;">
                <div style="margin-bottom: 0.5rem;">
                    <strong>Teks Asli ${index + 1}:</strong>
                    <p style="margin-top: 0.25rem; font-style: italic;">${item.original}</p>
                </div>
                <div>
                    <strong>Setelah Preprocessing:</strong>
                    <p style="margin-top: 0.25rem; color: var(--text-muted);">${item.preprocessed}</p>
                </div>
            </div>
        `;
    });

    previewContent.innerHTML = html;
    document.getElementById('preprocessPreview').classList.remove('hidden');
}

// Train Model
async function trainModel() {
    const hiddenLayersStr = document.getElementById('hiddenLayers').value;
    const hiddenLayers = hiddenLayersStr.split(',').map(n => parseInt(n.trim()));
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const activation = document.getElementById('activation').value;
    const testSize = parseInt(document.getElementById('testSize').value) / 100;

    showAlert('trainingAlert', 'Memulai training model...', 'info');
    document.getElementById('trainingProgress').classList.remove('hidden');

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 1, 95);
        updateProgress(progress, `Training epoch... ${progress}%`);
    }, (epochs * 100) / 95);  // Approximate timing

    try {
        const response = await fetch(`${API_URL}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                hidden_layers: hiddenLayers,
                learning_rate: learningRate,
                epochs: epochs,
                batch_size: batchSize,
                activation: activation,
                test_size: testSize,
                val_size: 0.1
            })
        });

        clearInterval(progressInterval);
        updateProgress(100, 'Training selesai!');

        const data = await response.json();

        if (data.success) {
            showAlert('trainingAlert', data.message, 'success');
            displayTrainingResults(data);
        } else {
            showAlert('trainingAlert', data.message, 'error');
        }
    } catch (error) {
        clearInterval(progressInterval);
        showAlert('trainingAlert', `Error: ${error.message}`, 'error');
    }
}

// Update Progress Bar
function updateProgress(percentage, text) {
    document.getElementById('progressFill').style.width = percentage + '%';
    document.getElementById('progressText').textContent = text;
}

// Display Training Results
function displayTrainingResults(data) {
    const history = data.history;
    const evaluation = data.evaluation;

    // Show charts section
    document.getElementById('chartsSection').classList.remove('hidden');

    // Create Loss Chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    if (lossChart) lossChart.destroy();

    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: history.loss.length }, (_, i) => i + 1),
            datasets: [
                {
                    label: 'Training Loss',
                    data: history.loss,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Validation Loss',
                    data: history.val_loss,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0'
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });

    // Create Accuracy Chart
    const accCtx = document.getElementById('accuracyChart').getContext('2d');
    if (accuracyChart) accuracyChart.destroy();

    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: history.accuracy.length }, (_, i) => i + 1),
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: history.accuracy,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Validation Accuracy',
                    data: history.val_accuracy,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0'
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    min: 0,
                    max: 1
                }
            }
        }
    });

    // Display Evaluation Stats
    displayEvaluation(evaluation, data.data_split);
}

// Display Evaluation
function displayEvaluation(evaluation, dataSplit) {
    const evaluationStats = document.getElementById('evaluationStats');

    evaluationStats.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Test Accuracy</div>
            <div class="stat-value">${(evaluation.test_accuracy * 100).toFixed(2)}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Train Samples</div>
            <div class="stat-value">${dataSplit.train_samples}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Val Samples</div>
            <div class="stat-value">${dataSplit.val_samples}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Test Samples</div>
            <div class="stat-value">${dataSplit.test_samples}</div>
        </div>
    `;

    // Confusion Matrix
    displayConfusionMatrix(evaluation.confusion_matrix, evaluation.classification_report);

    // Classification Report
    displayClassificationReport(evaluation.classification_report);

    document.getElementById('evaluationSection').classList.remove('hidden');
}

// Display Confusion Matrix
function displayConfusionMatrix(matrix, report) {
    const labels = Object.keys(report).filter(key => !['accuracy', 'macro avg', 'weighted avg'].includes(key));

    let html = '<table><thead><tr><th></th>';
    labels.forEach(label => {
        html += `<th>${label}</th>`;
    });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, i) => {
        html += `<tr><th>${labels[i]}</th>`;
        row.forEach(value => {
            html += `<td>${value}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';

    document.getElementById('confusionMatrix').innerHTML = html;
}

// Display Classification Report
function displayClassificationReport(report) {
    const labels = Object.keys(report).filter(key => !['accuracy', 'macro avg', 'weighted avg'].includes(key));

    let html = '<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">';
    html += '<thead><tr><th style="text-align: left; padding: 0.75rem; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--border);">Label</th>';
    html += '<th style="text-align: center; padding: 0.75rem; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--border);">Precision</th>';
    html += '<th style="text-align: center; padding: 0.75rem; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--border);">Recall</th>';
    html += '<th style="text-align: center; padding: 0.75rem; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--border);">F1-Score</th>';
    html += '<th style="text-align: center; padding: 0.75rem; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--border);">Support</th></tr></thead>';
    html += '<tbody>';

    labels.forEach(label => {
        const metrics = report[label];
        html += `<tr>
            <td style="padding: 0.75rem; border: 1px solid var(--border); background: rgba(99, 102, 241, 0.05);"><strong>${label}</strong></td>
            <td style="text-align: center; padding: 0.75rem; border: 1px solid var(--border); background: rgba(99, 102, 241, 0.05);">${metrics.precision.toFixed(3)}</td>
            <td style="text-align: center; padding: 0.75rem; border: 1px solid var(--border); background: rgba(99, 102, 241, 0.05);">${metrics.recall.toFixed(3)}</td>
            <td style="text-align: center; padding: 0.75rem; border: 1px solid var(--border); background: rgba(99, 102, 241, 0.05);">${metrics['f1-score'].toFixed(3)}</td>
            <td style="text-align: center; padding: 0.75rem; border: 1px solid var(--border); background: rgba(99, 102, 241, 0.05);">${metrics.support}</td>
        </tr>`;
    });

    html += '</tbody></table>';

    document.getElementById('reportContent').innerHTML = html;
}

// Predict Emotion
async function predictEmotion() {
    const text = document.getElementById('predictionText').value;

    if (!text.trim()) {
        showAlert('predictionAlert', 'Silakan masukkan teks terlebih dahulu', 'error');
        return;
    }

    showAlert('predictionAlert', 'Memprediksi emosi...', 'info');

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
            showAlert('predictionAlert', 'Prediksi berhasil!', 'success');
            displayPredictionResult(text, data);
        } else {
            showAlert('predictionAlert', data.message, 'error');
        }
    } catch (error) {
        showAlert('predictionAlert', `Error: ${error.message}`, 'error');
    }
}

// Display Prediction Result
function displayPredictionResult(text, data) {
    document.getElementById('originalText').textContent = text;
    document.getElementById('preprocessedText').textContent = data.preprocessed_text;
    document.getElementById('predictedEmotion').textContent = data.prediction.toUpperCase();
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(2)}%`;

    // Probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    let barsHTML = '';

    for (const [emotion, prob] of Object.entries(data.probabilities)) {
        const percentage = prob * 100;
        barsHTML += `
            <div class="confidence-bar">
                <div class="confidence-label">${emotion}</div>
                <div class="confidence-fill">
                    <div class="confidence-value" style="width: ${percentage}%">
                        ${percentage.toFixed(2)}%
                    </div>
                </div>
            </div>
        `;
    }

    probabilityBars.innerHTML = barsHTML;
    document.getElementById('predictionResult').classList.remove('hidden');
}

// Save Model
async function saveModel() {
    showAlert('trainingAlert', 'Menyimpan model...', 'info');

    try {
        const response = await fetch(`${API_URL}/api/save_model`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showAlert('trainingAlert', `Model berhasil disimpan! Files: ${data.files.model}`, 'success');
        } else {
            showAlert('trainingAlert', data.message, 'error');
        }
    } catch (error) {
        showAlert('trainingAlert', `Error: ${error.message}`, 'error');
    }
}

// Download Template
async function downloadTemplate() {
    try {
        window.open(`${API_URL}/api/download_template`, '_blank');
    } catch (error) {
        showAlert('uploadAlert', `Error: ${error.message}`, 'error');
    }
}

// Show Alert
function showAlert(elementId, message, type) {
    const alertEl = document.getElementById(elementId);
    alertEl.className = `alert alert-${type}`;

    const icon = {
        success: '✅',
        error: '❌',
        info: 'ℹ️'
    }[type] || 'ℹ️';

    alertEl.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    alertEl.classList.remove('hidden');
}

// File input change handler
document.getElementById('datasetFile').addEventListener('change', function (e) {
    const fileName = e.target.files[0]?.name || 'Pilih file atau drag & drop di sini';
    document.getElementById('fileName').textContent = fileName;
});
