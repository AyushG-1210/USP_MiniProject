let predictionChart;

async function queryPrediction(input) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
    });

    const data = await response.json();
    return Number(data.predicted_temperature);
}

function buildBatchInputsFromUser() {
    const x = parseFloat(document.getElementById('input-x').value);
    const y = parseFloat(document.getElementById('input-y').value);
    const t = parseFloat(document.getElementById('input-t').value);

    const stepCount = 10;
    const noteElement = document.getElementById('graph-note');

    if ([x, y, t].some(value => Number.isNaN(value) || value < 0 || value > 1)) {
        if (noteElement) {
            noteElement.textContent = 'Please enter x, y, and t values between 0 and 1.';
        }
        return [];
    }

    const batchInputs = [];

    for (let i = 0; i < stepCount; i++) {
        const ratio = i / (stepCount - 1);

        batchInputs.push({
            x: x + (1.0 - x) * ratio,
            y: y + (1.0 - y) * ratio,
            t: t + (1.0 - t) * ratio
        });
    }

    if (noteElement) {
        noteElement.textContent =
            `Graphing from your inputs (${x.toFixed(2)}, ${y.toFixed(2)}, ${t.toFixed(2)}) and stepping them evenly to 1.0 for x, y, and t.`;
    }

    return batchInputs;
}

async function runBatchPrediction() {
    const batchInputs = buildBatchInputsFromUser();

    if (batchInputs.length === 0) {
        return;
    }

    const results = [];

    for (let i = 0; i < batchInputs.length; i++) {
        const value = await queryPrediction(batchInputs[i]);
        results.push({
            step: i + 1,
            input: batchInputs[i],
            output: value
        });
    }

    renderChart(results);
}

function renderChart(results) {
    const labels = results.map(item => `Step ${item.step}`);
    const values = results.map(item => item.output);

    const ctx = document.getElementById('predictionChart');

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Predicted Temperature',
                data: values,
                borderColor: '#0066cc',
                backgroundColor: 'rgba(0, 102, 204, 0.15)',
                borderWidth: 2,
                tension: 0.25,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

window.addEventListener('DOMContentLoaded', () => {
    const graphButton = document.getElementById('graph-button');
    if (graphButton) {
        graphButton.addEventListener('click', runBatchPrediction);
    }
});