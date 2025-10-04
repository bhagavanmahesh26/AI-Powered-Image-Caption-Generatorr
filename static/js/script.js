const form = document.getElementById("caption-form");
const imageInput = document.getElementById("image-input");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("result");
const previewImage = document.getElementById("preview-image");
const captionText = document.getElementById("caption-text");

const resetResult = () => {
    statusEl.textContent = "";
    captionText.textContent = "";
    resultSection.hidden = true;
};

const showStatus = (message, isError = false) => {
    statusEl.textContent = message;
    statusEl.classList.toggle("error", isError);
};

imageInput.addEventListener("change", () => {
    resetResult();

    const file = imageInput.files?.[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        resultSection.hidden = false;
    };
    reader.readAsDataURL(file);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resetResult();

    const file = imageInput.files?.[0];
    if (!file) {
        showStatus("Please select an image to upload.", true);
        return;
    }

    showStatus("Uploading image and generating caption...");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/caption", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Failed to generate caption.");
        }

        captionText.textContent = data.caption;
        resultSection.hidden = false;
        showStatus("Caption generated successfully!");
    } catch (error) {
        showStatus(error.message, true);
    }
});
