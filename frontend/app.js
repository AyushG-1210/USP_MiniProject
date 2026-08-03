(function () {
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

        return {
            material,
            lengthScale,
            baseline,
            deltaTemp
        };
    }

    function populateMaterialOptions() {
        const materialSelect = document.getElementById('material-select');
        if (!materialSelect) {
            return;
        }

        materialSelect.querySelectorAll('option').forEach((option) => {
            const material = MATERIALS[option.value];
            if (material) {
                option.textContent = `${material.label} (α = ${material.diffusivity.toExponential(2)} m²/s)`;
            }
        });
    }

    function convertToPhysicalForPrediction(rawValue, x, y, t, settings) {
        const material = MATERIALS[settings.material] ?? MATERIALS.copper;
        const tau = (settings.lengthScale ** 2) / material.diffusivity;
        const tReal = tau * t;
        const xReal = settings.lengthScale * x;
        const yReal = settings.lengthScale * y;
        const uReal = settings.baseline + settings.deltaTemp * rawValue;

        return {
            rawValue,
            xReal,
            yReal,
            tReal,
            uReal
        };
    }

    function getResultExplanation(physical, settings) {
        const material = MATERIALS[settings.material] ?? MATERIALS.copper;
        return [
            `Model output: ${physical.rawValue.toFixed(4)} (dimensionless)`,
            `Converted temperature: ${physical.uReal.toFixed(4)} °C`,
            `Physical time scale: ${physical.tReal.toFixed(2)} s`,
            `Spatial position: x = ${(physical.xReal).toFixed(3)} m, y = ${(physical.yReal).toFixed(3)} m`,
            `Plate length L = ${settings.lengthScale.toFixed(3)} m`,
            `Material = ${material.label}`
        ].join(' · ');
    }

    async function handlePredict(event) {
        if (event) event.preventDefault();

        const x = parseFloat(document.getElementById('input-x').value);
        const y = parseFloat(document.getElementById('input-y').value);
        const t = parseFloat(document.getElementById('input-t').value);

        const errorContainer = document.getElementById('error-display');
        const resultDisplay = document.getElementById('result-display');
        const resultMeta = document.getElementById('result-meta');
        const resultBox = document.getElementById('resultBox');
        const button = document.getElementById('predict-button');

        if ([x, y, t].some(value => Number.isNaN(value) || value < 0 || value > 1)) {
            errorContainer.textContent = 'Please enter x, y, and t values between 0 and 1.';
            resultDisplay.textContent = 'No result';
            resultMeta.textContent = 'Raw model output shown in dimensionless units';
            resultBox.style.display = 'block';
            return;
        }

        button.disabled = true;
        button.textContent = 'Computing...';
        resultDisplay.textContent = 'Loading...';
        resultMeta.textContent = 'Converting to physical units...';
        errorContainer.textContent = '';
        resultBox.style.display = 'block';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x, y, t })
            });

            const rawText = await response.text();

            if (!response.ok) {
                let message = 'Unable to compute prediction. Please try again.';
                try {
                    const parsed = JSON.parse(rawText);
                    if (parsed?.detail) {
                        message = 'Please enter x, y, and t values between 0 and 1.';
                    }
                } catch {
                    // ignore parse errors
                }
                throw new Error(message);
            }

            const data = JSON.parse(rawText);
            const rawValue = Number(data.predicted_temperature);
            const settings = getPhysicalSettings();
            const physical = convertToPhysicalForPrediction(rawValue, x, y, t, settings);

            resultDisplay.textContent = `${physical.uReal.toFixed(4)} °C`;
            resultMeta.textContent = getResultExplanation(physical, settings);

        } catch (error) {
            resultDisplay.textContent = 'No result';
            resultMeta.textContent = 'Raw model output shown in dimensionless units';
            errorContainer.textContent = error.message;
        } finally {
            button.disabled = false;
            button.textContent = 'Compute Inference';
        }
    }

    window.handlePredict = handlePredict;

    function initializePredictUI() {
        const button = document.getElementById('predict-button');
        const materialSelect = document.getElementById('material-select');

        populateMaterialOptions();

        if (button) {
            button.addEventListener('click', handlePredict);
        }

        if (materialSelect) {
            materialSelect.addEventListener('change', async () => {
                const resultBox = document.getElementById('resultBox');
                const errorContainer = document.getElementById('error-display');
                if (resultBox) {
                    resultBox.style.display = 'block';
                }
                if (errorContainer) {
                    errorContainer.textContent = 'Material updated. Refreshing the physical conversion...';
                }
                await handlePredict();
            });
        }
    }

    initializePredictUI();
})();