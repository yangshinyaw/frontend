let uploadedImage = null;
let detectedWords = [];

// Listen for file upload
document.getElementById('fileInput').addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        uploadedImage = img;
        detectedWords = []; // Reset detected words
    };

    img.src = URL.createObjectURL(file);
}

// Handle the "Detect Words" button click
document.getElementById('detectButton').addEventListener('click', handleDetection);

async function handleDetection() {
    if (!uploadedImage) return;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(uploadedImage, 0, 0);

    preprocessImage(canvas);

    // Perform CCA + Contour Detection
    const contours = detectContours(canvas);

    // Group contours into lines and merge nearby contours into words
    const mergedWords = groupContoursIntoLinesAndWords(contours, 70, 40);

    detectedWords = mergedWords.map(contour => ({
        bbox: {
            x0: contour.x0,
            y0: contour.y0,
            x1: contour.x1,
            y1: contour.y1
        },
        text: 'Detected Word'
    }));

    drawDetectedWords(ctx);
}

// Preprocess the image (grayscale + binarization)
function preprocessImage(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Convert to grayscale
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = data[i + 1] = data[i + 2] = avg;
    }

    // Apply binarization threshold
    const threshold = 128;
    for (let i = 0; i < data.length; i += 4) {
        const avg = data[i];
        const value = avg > threshold ? 255 : 0;
        data[i] = data[i + 1] = data[i + 2] = value;
    }

    ctx.putImageData(imageData, 0, 0);
}

// Detect contours using CCA (Connected Component Analysis)
function detectContours(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    const visited = Array(height).fill().map(() => Array(width).fill(false));
    const contours = [];

    function bfs(x, y) {
        const queue = [[x, y]];
        let minX = x, minY = y, maxX = x, maxY = y;

        visited[y][x] = true;

        while (queue.length > 0) {
            const [cx, cy] = queue.shift();
            minX = Math.min(minX, cx);
            minY = Math.min(minY, cy);
            maxX = Math.max(maxX, cx);
            maxY = Math.max(maxY, cy);

            const neighbors = [
                [cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]
            ];

            neighbors.forEach(([nx, ny]) => {
                if (nx >= 0 && ny >= 0 && nx < width && ny < height && !visited[ny][nx]) {
                    const index = (ny * width + nx) * 4;
                    if (data[index] === 0) { // Check if it's part of the foreground
                        visited[ny][nx] = true;
                        queue.push([nx, ny]);
                    }
                }
            });
        }

        return { x0: minX, y0: minY, x1: maxX, y1: maxY };
    }

    // Loop through each pixel to find connected components
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            if (data[index] === 0 && !visited[y][x]) { // Black pixel and not visited
                const contour = bfs(x, y);
                // Filter out very small contours (likely noise)
                const contourWidth = contour.x1 - contour.x0;
                const contourHeight = contour.y1 - contour.y0;
                if (contourWidth > 5 && contourHeight > 5) { // Avoid noise
                    contours.push(contour);
                }
            }
        }
    }

    return contours;
}

// Group contours into lines and merge nearby contours into words
function groupContoursIntoLinesAndWords(contours, horizontalThreshold, verticalThreshold) {
    const lines = [];

    contours.forEach(contour => {
        let addedToLine = false;

        for (const line of lines) {
            const [firstInLine] = line;
            if (Math.abs(contour.y0 - firstInLine.y0) <= verticalThreshold) {
                line.push(contour);
                addedToLine = true;
                break;
            }
        }

        if (!addedToLine) {
            lines.push([contour]);
        }
    });

    const mergedWords = [];

    lines.forEach(line => {
        line.sort((a, b) => a.x0 - b.x0);

        let currentWord = null;

        line.forEach(contour => {
            if (!currentWord) {
                currentWord = { ...contour };
            } else {
                const distance = contour.x0 - currentWord.x1;
                if (distance <= horizontalThreshold) {
                    currentWord.x1 = Math.max(currentWord.x1, contour.x1);
                    currentWord.y0 = Math.min(currentWord.y0, contour.y0);
                    currentWord.y1 = Math.max(currentWord.y1, contour.y1);
                } else {
                    mergedWords.push(currentWord);
                    currentWord = { ...contour };
                }
            }
        });

        if (currentWord) {
            mergedWords.push(currentWord);
        }
    });

    return mergedWords;
}

// Draw the detected words with red bounding boxes
function drawDetectedWords(ctx) {
    detectedWords.forEach(word => {
        ctx.strokeStyle = 'red';  // Red box color
        ctx.lineWidth = 2;  // Line thickness for the red boxes
        ctx.strokeRect(word.bbox.x0, word.bbox.y0, word.bbox.x1 - word.bbox.x0, word.bbox.y1 - word.bbox.y0);
    });
}

// Handle the "Extract Words" button click for TrOCR extraction
document.getElementById('extractButton').addEventListener('click', async () => {
    if (!uploadedImage) return;

    const canvas = document.getElementById('canvas');
    const formData = new FormData();

    for (let i = 0; i < detectedWords.length; i++) {
        const word = detectedWords[i];
        const bbox = word.bbox;

        const croppedCanvas = document.createElement('canvas');
        const croppedCtx = croppedCanvas.getContext('2d');
        const width = bbox.x1 - bbox.x0;
        const height = bbox.y1 - bbox.y0;
        croppedCanvas.width = width;
        croppedCanvas.height = height;
        croppedCtx.drawImage(canvas, bbox.x0, bbox.y0, width, height, 0, 0, width, height);

        await new Promise(resolve => {
            croppedCanvas.toBlob(blob => {
                formData.append('files', blob, `word_${i}.png`);
                resolve();
            }, 'image/png');
        });
    }

    const response = await fetch('https://backend-pw0r.onrender.com/extract', {  // Updated URL
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const data = await response.json();
        const predictedWords = document.getElementById('predictedWords');
        data.texts.forEach(text => {
            const wordItem = document.createElement("li");
            wordItem.textContent = text;  // Display extracted text
            predictedWords.appendChild(wordItem);
        });
    } else {
        console.error('Error extracting text:', response.statusText);
    }
});
