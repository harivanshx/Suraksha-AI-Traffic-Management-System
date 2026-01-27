// AI Traffic Management System - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function () {
    const directions = ['NORTH', 'SOUTH', 'EAST', 'WEST'];
    const uploadedFiles = {};
    let processingMode = null;

    // ==================== BULK UPLOAD FUNCTIONALITY ====================
    const bulkSelectBtn = document.getElementById('bulk-select-btn');
    const bulkFileInput = document.getElementById('bulk-file-input');
    const bulkUploadStatus = document.getElementById('bulk-upload-status');
    const bulkPreview = document.getElementById('bulk-preview');

    // Trigger file input when button is clicked
    bulkSelectBtn.addEventListener('click', function () {
        bulkFileInput.click();
    });

    // Handle bulk file selection
    bulkFileInput.addEventListener('change', async function (e) {
        const files = Array.from(e.target.files);

        // Validate file count
        if (files.length !== 4) {
            showBulkStatus(`Please select exactly 4 images. You selected ${files.length}.`, 'error');
            bulkFileInput.value = ''; // Reset input
            return;
        }

        // Validate all are images
        const imageExts = ['jpg', 'jpeg', 'png', 'bmp'];
        const invalidFiles = files.filter(file => {
            const ext = file.name.toLowerCase().split('.').pop();
            return !imageExts.includes(ext);
        });

        if (invalidFiles.length > 0) {
            showBulkStatus('All files must be images (JPG, PNG, BMP).', 'error');
            bulkFileInput.value = '';
            return;
        }

        // Set processing mode to image
        if (processingMode && processingMode !== 'image') {
            showBulkStatus(`Cannot upload images. Current mode is ${processingMode}.`, 'error');
            bulkFileInput.value = '';
            return;
        }
        processingMode = 'image';

        // Show preview and let user assign directions
        showBulkPreview(files);
    });

    function showBulkPreview(files) {
        bulkPreview.innerHTML = '';
        bulkPreview.style.display = 'grid';

        const assignmentDiv = document.createElement('div');
        assignmentDiv.className = 'bulk-assignment';
        assignmentDiv.innerHTML = '<h4>Assign Images to Directions:</h4>';

        files.forEach((file, index) => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'bulk-item';

            // Create image preview
            const imgPreview = document.createElement('img');
            imgPreview.className = 'bulk-img-preview';
            const reader = new FileReader();
            reader.onload = function (e) {
                imgPreview.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // Create direction selector
            const select = document.createElement('select');
            select.className = 'bulk-direction-select';
            select.dataset.fileIndex = index;

            directions.forEach((dir, dirIndex) => {
                const option = document.createElement('option');
                option.value = dir;
                option.textContent = `${dir} ${getDirectionArrow(dir)}`;
                if (index === dirIndex) {
                    option.selected = true; // Auto-assign in order
                }
                select.appendChild(option);
            });

            const fileName = document.createElement('div');
            fileName.className = 'bulk-file-name';
            fileName.textContent = file.name;

            itemDiv.appendChild(imgPreview);
            itemDiv.appendChild(select);
            itemDiv.appendChild(fileName);
            assignmentDiv.appendChild(itemDiv);
        });

        // Add upload button
        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'btn-primary bulk-upload-btn';
        uploadBtn.innerHTML = '<span class="btn-icon">⬆️</span> Upload All 4 Images';
        uploadBtn.addEventListener('click', () => handleBulkUpload(files));

        bulkPreview.appendChild(assignmentDiv);
        bulkPreview.appendChild(uploadBtn);

        showBulkStatus('✓ 4 images selected. Review assignments and click "Upload All".', 'success');
    }

    async function handleBulkUpload(files) {
        const selects = document.querySelectorAll('.bulk-direction-select');
        const assignments = {};

        // Build assignment map
        selects.forEach(select => {
            const fileIndex = parseInt(select.dataset.fileIndex);
            const direction = select.value;
            assignments[direction] = files[fileIndex];
        });

        // Check for duplicate assignments
        if (Object.keys(assignments).length !== 4) {
            showBulkStatus('Each direction must have a unique image assigned!', 'error');
            return;
        }

        // Disable button and show progress
        const uploadBtn = bulkPreview.querySelector('.bulk-upload-btn');
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';

        showBulkStatus('⏳ Uploading all 4 images...', 'info');

        try {
            // Upload all files in parallel
            const uploadPromises = Object.entries(assignments).map(([direction, file]) => {
                return uploadBulkFile(file, direction);
            });

            const results = await Promise.all(uploadPromises);

            // Check if all succeeded
            const allSuccess = results.every(r => r.success);

            if (allSuccess) {
                showBulkStatus('✓ All 4 images uploaded successfully!', 'success');

                // Update individual drop zones to show uploaded state
                results.forEach(result => {
                    const direction = result.direction;
                    const dirLower = direction.toLowerCase();
                    const dropZone = document.getElementById(`drop-${dirLower}`);
                    const fileInfo = dropZone.querySelector('.file-info');
                    const statusDiv = document.getElementById(`status-${dirLower}`);

                    dropZone.classList.add('uploaded');
                    fileInfo.textContent = `✓ ${result.filename}`;
                    fileInfo.style.color = '#10b981';
                    showStatus(statusDiv, `✓ ${result.filename} uploaded (bulk)`, 'success');

                    uploadedFiles[direction] = result.filename;
                });

                // Update run button
                updateRunButton();

                // Hide preview after 2 seconds
                setTimeout(() => {
                    bulkPreview.style.display = 'none';
                    bulkFileInput.value = '';
                }, 2000);
            } else {
                showBulkStatus('⚠️ Some uploads failed. Check individual statuses.', 'error');
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<span class="btn-icon">⬆️</span> Retry Upload';
            }

        } catch (error) {
            console.error('Bulk upload error:', error);
            showBulkStatus('✗ Upload failed: ' + error.message, 'error');
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<span class="btn-icon">⬆️</span> Retry Upload';
        }
    }

    async function uploadBulkFile(file, direction) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('direction', direction);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            return {
                success: true,
                direction: direction,
                filename: data.filename
            };
        } catch (error) {
            // Update individual status div
            const dirLower = direction.toLowerCase();
            const statusDiv = document.getElementById(`status-${dirLower}`);
            showStatus(statusDiv, '✗ ' + error.message, 'error');

            return {
                success: false,
                direction: direction,
                error: error.message
            };
        }
    }

    function showBulkStatus(message, type) {
        bulkUploadStatus.textContent = message;
        bulkUploadStatus.className = 'bulk-status ' + type;
        bulkUploadStatus.style.display = 'block';
    }

    function getDirectionArrow(direction) {
        const arrows = {
            'NORTH': '↑',
            'SOUTH': '↓',
            'EAST': '→',
            'WEST': '←'
        };
        return arrows[direction] || '';
    }

    // ==================== INDIVIDUAL UPLOAD FUNCTIONALITY ====================
    // Initialize drag-and-drop for all directions
    directions.forEach(direction => {
        const dirLower = direction.toLowerCase();
        const dropZone = document.getElementById(`drop-${dirLower}`);
        const fileInput = document.getElementById(`file-${dirLower}`);
        const status = document.getElementById(`status-${dirLower}`);

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', function (e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0) {
                handleFile(files[0], direction, dropZone, fileInput, status);
            }
        }, false);

        // Handle file input change (click to upload)
        fileInput.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0], direction, dropZone, fileInput, status);
            }
        });

        // Click on drop zone to trigger file input
        dropZone.addEventListener('click', function (e) {
            // Don't open file explorer if already uploaded
            if (!dropZone.classList.contains('uploaded')) {
                // Prevent double-trigger: don't click if the target is the file input itself
                if (e.target !== fileInput && !fileInput.contains(e.target)) {
                    fileInput.click();
                }
            }
        });
    });

    // Run simulation button
    const runButton = document.getElementById('run-simulation');
    const processingInfo = document.getElementById('processing-info');

    runButton.addEventListener('click', async function () {
        // Disable button and show processing
        runButton.disabled = true;
        processingInfo.style.display = 'flex';

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Processing failed');
            }

            if (data.success) {
                // Redirect to results page
                window.location.href = data.redirect;
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing simulation: ' + error.message);
            runButton.disabled = false;
            processingInfo.style.display = 'none';
        }
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleFile(file, direction, dropZone, fileInput, statusDiv) {
        // Validate file type
        const fileName = file.name.toLowerCase();
        const ext = fileName.split('.').pop();

        const videoExts = ['mp4', 'avi', 'mov', 'mkv'];
        const imageExts = ['jpg', 'jpeg', 'png', 'bmp'];

        let fileType = null;
        if (videoExts.includes(ext)) {
            fileType = 'video';
        } else if (imageExts.includes(ext)) {
            fileType = 'image';
        } else {
            showStatus(statusDiv, 'Invalid file type. Please upload a video (MP4, AVI, MOV, MKV) or image (JPG, PNG, BMP).', 'error');
            return;
        }

        // Check processing mode consistency
        if (processingMode && processingMode !== fileType) {
            showStatus(statusDiv, `Please upload all ${processingMode} files. Cannot mix videos and images.`, 'error');
            return;
        }

        // Set processing mode
        if (!processingMode) {
            processingMode = fileType;
        }

        // Upload file
        uploadFile(file, direction, dropZone, fileInput, statusDiv);
    }

    async function uploadFile(file, direction, dropZone, fileInput, statusDiv) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('direction', direction);

        // Show uploading status
        showStatus(statusDiv, '⏳ Uploading...', 'info');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            // Update UI
            uploadedFiles[direction] = data.filename;
            dropZone.classList.add('uploaded');

            // Update file info
            const fileInfo = dropZone.querySelector('.file-info');
            fileInfo.textContent = `✓ ${data.filename}`;
            fileInfo.style.color = '#10b981';

            showStatus(statusDiv, `✓ ${data.filename} uploaded successfully`, 'success');

            // Check if all directions have files
            updateRunButton();

        } catch (error) {
            console.error('Upload error:', error);
            showStatus(statusDiv, '✗ ' + error.message, 'error');
            dropZone.classList.remove('uploaded');
        }
    }

    function showStatus(statusDiv, message, type) {
        statusDiv.textContent = message;
        statusDiv.className = 'upload-status ' + type;

        if (type === 'error') {
            // Auto-hide error after 5 seconds
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    }

    function updateRunButton() {
        const runButton = document.getElementById('run-simulation');
        const uploadCount = Object.keys(uploadedFiles).length;

        if (uploadCount >= 1) {
            runButton.disabled = false;
            runButton.textContent = `▶️ Run Simulation (${uploadCount}/4 directions)`;
        } else {
            runButton.disabled = true;
            runButton.innerHTML = '<span class="btn-icon">▶️</span> Run Simulation';
        }
    }
});
