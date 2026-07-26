async function handlePredict(event) {
    event.preventDefault();

    const x = parseFloat(document.getElementById('input-x').value);
    const y = parseFloat(document.getElementById('input-y').value);
    const t = parseFloat(document.getElementById('input-t').value);

    const errorContainer = document.getElementById('error-display'); // Your UI error element
    const resultDisplay = document.getElementById('result-display');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, t })
        });

        if (!response.ok) {
            // Catches Pydantic 422 validation errors or server issues
            throw new Error("All input values (x, y, t) must be between 0 and 1.");
        }

        const data = await response.json();
        
        // Clear errors and show prediction
        errorContainer.textContent = '';
        resultDisplay.textContent = `Predicted Temperature: ${data.predicted_temperature.toFixed(4)} °C`;

    } catch (error) {
        // Render user-friendly error popup/text
        resultDisplay.textContent = '';
        errorContainer.textContent = error.message;
    }
}