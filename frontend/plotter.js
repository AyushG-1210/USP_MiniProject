let predictionChart;

const MATERIALS = {
    copper: { diffusivity: 1.1e-4, label: 'Copper' },
    aluminum: { diffusivity: 8.6e-5, label: 'Aluminum' },
    steel: { diffusivity: 1.5e-6, label: 'Steel' },
    silicon: { diffusivity: 8.8e-5, label: 'Silicon' }
};

function getPhysicalSettings() {
    const material = document.getElementById('material-select').value;
    const lengthScale = parseFloat(document.getElementById('plate-size').value);
    const baseline = parseFloat(document.getElementById('baseline-temp').value);
    const deltaTemp = parseFloat(document.getElementById('delta-temp').value);

    return { material, lengthScale, baseline, deltaTemp };
}

function convertToPhysicalForChart(rawValue, x, y, t, settings) {
    const material = MATERIALS[settings.material] ?? MATERIALS.copper;
    const tau = (settings.lengthScale ** 2) / material.diffusivity;
    const tReal = tau * t;
    const uReal = settings.baseline + settings.deltaTemp * rawValue;

    return {
        rawValue,
        tReal,
        uReal
    };
}

function describeSelectedPhysics(settings) {
    const material = MATERIALS[settings.material] ?? MATERIALS.copper;
    const tau = (settings.lengthScale ** 2) / material.diffusivity;
    return `Using ${material.label}, plate length L = ${settings.lengthScale.toFixed(3)} m, base temperature u_ref = ${settings.baseline.toFixed(1)} °C, temperature scale ΔT = ${settings.deltaTemp.toFixed(1)} °C, so one model-time unit corresponds to about ${tau.toFixed(2)} s in physical time.`;
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
            x,
            y,
            t: t + (1.0 - t) * ratio
        });
    }

    const settings = getPhysicalSettings();

    if (noteElement) {
        noteElement.textContent =
            `The graph follows a time sweep from your entered model-time point up to the end of the normalized domain. ` +
            `Here x and y stay fixed, while t changes from ${t.toFixed(4)} to 1.0 in the model coordinate system. ` +
            describeSelectedPhysics(settings);
    }

    return batchInputs;
}

async function runBatchPrediction() {
    const batchInputs = buildBatchInputsFromUser();

    if (batchInputs.length === 0) {
        return;
    }

    try {
        const response = await fetch('/predict-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: batchInputs })
        });

        const rawText = await response.text();

        if (!response.ok) {
            throw new Error(`Batch computation failed: ${rawText}`);
        }

        const data = JSON.parse(rawText);
        const predictions = Array.isArray(data.predictions) ? data.predictions : [];

        const settings = getPhysicalSettings();
        const results = predictions.map((value, i) => {
            const current = batchInputs[i];
            const physical = convertToPhysicalForChart(Number(value), current.x, current.y, current.t, settings);
            return {
                step: i + 1,
                input: current,
                output: physical.uReal
            };
        });

        renderChart(results);

    } catch (error) {
        const noteElement = document.getElementById('graph-note');
        if (noteElement) {
            noteElement.textContent = error.message;
        }
    }
}

function renderChart(results) {
    const ctx = document.getElementById('predictionChart');

    if (!ctx) {
        return;
    }

    const settings = getPhysicalSettings();
    const material = MATERIALS[settings.material] ?? MATERIALS.copper;
    const tau = (settings.lengthScale ** 2) / material.diffusivity;

    const labels = results.map((item) => {
        const t = item.input.t;
        return `${t.toFixed(2)} → ${(tau * t).toFixed(0)} s`;
    });
    const values = results.map(item => item.output);

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Predicted temperature u(x, y, t) in °C',
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
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(4)} °C`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Model time t → physical time' },
                    ticks: {
                        autoSkip: true,
                        maxTicksLimit: 6,
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    beginAtZero: false,
                    title: { display: true, text: 'Temperature (°C)' },
                    ticks: {
                        precision: 1
                    }
                }
            }
        }
    });
}

window.runBatchPrediction = runBatchPrediction;

function initializeGraphUI() {
    const graphButton = document.getElementById('graph-button');
    if (graphButton) {
        graphButton.addEventListener('click', runBatchPrediction);
    }
}

initializeGraphUI();