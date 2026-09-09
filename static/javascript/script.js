const fileInput = document.getElementById("csvFile");

fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];

    if (file && file.name.endsWith(".csv")) {
        alert("CSV file has been selected: " + file.name);
        handleCsvFile(file);
    } else {
        alert("Please select a CSV file.");
    }
});

async function handleCsvFile(file) {

    const formData = new FormData();
    formData.append("csv_file", file);

    try {
        const response = await fetch("/upload-csv", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        console.log(result);

        window.location.href = result.redirect_url;

    } catch (error) {
        console.error(error);
        alert("Upload failed");
    }
}