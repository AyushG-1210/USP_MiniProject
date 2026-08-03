async function handlePredict(event) {
    if (event) event.preventDefault();

    const x = parseFloat(document.getElementById('input-x').value);
    const y = parseFloat(document.getElementById('input-y').value);
    const t = parseFloat(document.getElementById('input-t').value);

    const errorContainer = document.getElementById('error-display');
    const resultDisplay = document.getElementById('result-display');
    const resultBox = document.getElementById('resultBox');
    const button = document.getElementById('predict-button');

    if ([x, y, t].some(value => Number.isNaN(value) || value < 0 || value > 1)) {
        errorContainer.textContent = 'Please enter x, y, and t values between 0 and 1.';
        resultDisplay.textContent = 'No result';
        resultBox.style.display = 'block';
        return;
    }

    button.disabled = true;
    button.textContent = 'Computing...';
    resultDisplay.textContent = 'Loading...';
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
        resultDisplay.textContent = `${Number(data.predicted_temperature).toFixed(4)} °C`;

    } catch (error) {
        resultDisplay.textContent = 'No result';
        errorContainer.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = 'Compute Inference';
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('predict-button');
    if (button) {
        button.addEventListener('click', handlePredict);
    }
});