const predictForm = document.getElementById("predictForm");
const resultsStatus = document.getElementById("resultsStatus");
const resultsTable = document.getElementById("resultsTable");

predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) {
    resultsStatus.textContent = "Selecciona un CSV primero.";
    return;
  }

  resultsStatus.textContent = "Procesando... esto puede tardar.";
  resultsTable.innerHTML = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload-and-predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      resultsStatus.textContent = data.error || "Error al procesar.";
      return;
    }

    resultsStatus.textContent = `Predicciones: ${data.predictions.length}`;
    const rows = data.predictions
      .slice(0, 20)
      .map(
        (p) =>
          `<tr><td>${p.unique_id}</td><td>${p.prediction_next_month.toFixed(
            2
          )}</td></tr>`
      )
      .join("");

    resultsTable.innerHTML = `
      <table>
        <thead><tr><th>unique_id</th><th>prediction_next_month</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="muted">Mostrando hasta 20 filas.</p>
    `;
  } catch (err) {
    resultsStatus.textContent = "Error de red.";
  }
});
