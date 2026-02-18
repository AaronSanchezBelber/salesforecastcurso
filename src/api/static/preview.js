const fileInput = document.getElementById("csvFile");
const previewStatus = document.getElementById("previewStatus");
const previewColumns = document.getElementById("previewColumns");
const previewSample = document.getElementById("previewSample");

function parseCSVLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && next === '"') {
      current += '"';
      i++;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  result.push(current);
  return result;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    previewStatus.textContent = "No hay archivo cargado.";
    previewColumns.innerHTML = "";
    previewSample.innerHTML = "";
    return;
  }

  previewStatus.textContent = `Archivo: ${file.name} (${(file.size / 1024).toFixed(
    1
  )} KB)`;

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length === 0) {
      previewColumns.innerHTML = "";
      previewSample.innerHTML = "";
      return;
    }

    const header = parseCSVLine(lines[0]);
    previewColumns.innerHTML = header
      .map((col) => `<span class="pill">${col}</span>`)
      .join("");

    if (lines.length > 1) {
      const sample = parseCSVLine(lines[1]);
      previewSample.innerHTML =
        "<h3>Ejemplo fila 1</h3>" +
        "<ul>" +
        sample.map((v, i) => `<li><b>${header[i] || "col"}:</b> ${v}</li>`).join("") +
        "</ul>";
    } else {
      previewSample.innerHTML = "<p class='muted'>No hay filas de datos.</p>";
    }
  };
  reader.readAsText(file);
});
