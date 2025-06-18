// Global variables to store segment context for navigation
let currentSegmentsList = [];
let currentSegmentFolderGlobal = '';
let currentSegmentPageIdGlobal = '';
let currentSegmentDisplayIndex = -1;

// Core initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('OCR page ready');
    
    initializeEventListeners();
    checkInitialSegments();
});

function initializeEventListeners() {
    const prevBtn = document.getElementById('prev-segment-btn');
    const nextBtn = document.getElementById('next-segment-btn');
    const viewJsonBtn = document.getElementById('view-segment-json-btn');
    const closeJsonBtn = document.getElementById('close-segment-json-btn');
    const previewImage = document.getElementById('preview-image');
    const segmentSelect = document.getElementById('segment-select');

    if (previewImage) {
        previewImage.addEventListener('load', initializePreviewOverlay);
    }

    if (prevBtn) {
        prevBtn.onclick = function() {
            if (currentSegmentDisplayIndex > 0) {
                displaySegmentByIndex(currentSegmentDisplayIndex - 1);
            }
        };
    }

    if (nextBtn) {
        nextBtn.onclick = function() {
            if (currentSegmentDisplayIndex < currentSegmentsList.length - 1) {
                displaySegmentByIndex(currentSegmentDisplayIndex + 1);
            }
        };
    }

    if (viewJsonBtn) {
        viewJsonBtn.onclick = function() {
            toggleSegmentJson(true);
        };
    }
    
    if (closeJsonBtn) {
        closeJsonBtn.onclick = function() {
            toggleSegmentJson(false);
        };
    }

    if (segmentSelect) {
        segmentSelect.onchange = function() {
            const selectedIndex = parseInt(this.value, 10);
            if (!isNaN(selectedIndex)) {
                displaySegmentByIndex(selectedIndex, true);
            }
        };
    }
}

function checkInitialSegments() {
    const selectedPng = new URLSearchParams(window.location.search).get('selected_png');
    const selectedFolder = new URLSearchParams(window.location.search).get('selected_folder');
    
    if (selectedPng && selectedFolder) {
        const pageId = selectedPng.split('.')[0];
        checkForExistingSegments(selectedFolder, pageId);
    }
}

function initializePreviewOverlay() {
    const previewImage = document.getElementById('preview-image');
    const overlay = document.getElementById('preview-overlay');
    
    if (previewImage && overlay) {
        console.log('Initializing preview overlay');
        
        // Set SVG viewport to match original image dimensions
        overlay.setAttribute('viewBox', `0 0 ${previewImage.naturalWidth} ${previewImage.naturalHeight}`);
        
        // Update overlay dimensions when image is resized
        const updateOverlaySize = () => {
            const rect = previewImage.getBoundingClientRect();
            overlay.style.width = `${rect.width}px`;
            overlay.style.height = `${rect.height}px`;
        };

        // Initial size update
        updateOverlaySize();
        
        // Window resize handler
        window.addEventListener('resize', updateOverlaySize);

        // Add resize observer to handle container/window resizing
        if (typeof ResizeObserver !== 'undefined') {
            const resizeObserver = new ResizeObserver(updateOverlaySize);
            resizeObserver.observe(previewImage.parentElement);
        }
    }
}

// Clear any existing highlight
function clearSegmentHighlight() {
    const overlay = document.getElementById('preview-overlay');
    if (overlay) {
        overlay.innerHTML = '';
    }
}

// Highlight a segment with the given coordinates
function highlightSegment(coords) {
    const overlay = document.getElementById('preview-overlay');
    const previewImage = document.getElementById('preview-image');
    
    if (!overlay || !previewImage || !Array.isArray(coords) || coords.length !== 4) {
        console.warn('[highlightSegment] Invalid parameters:', { overlay: !!overlay, coords });
        return;
    }

    // Clear existing highlights
    clearSegmentHighlight();

    // Get both natural (original) and displayed dimensions
    const imageNaturalWidth = previewImage.naturalWidth;
    const imageNaturalHeight = previewImage.naturalHeight;
    const imageDisplayRect = previewImage.getBoundingClientRect();
    const imageDisplayWidth = imageDisplayRect.width;
    const imageDisplayHeight = imageDisplayRect.height;

    // Calculate scaling factors
    const scaleX = imageDisplayWidth / imageNaturalWidth;
    const scaleY = imageDisplayHeight / imageNaturalHeight;

    // Log dimensions and scaling for debugging
    console.log('[highlightSegment] Dimensions:', {
        natural: { width: imageNaturalWidth, height: imageNaturalHeight },
        display: { width: imageDisplayWidth, height: imageDisplayHeight },
        scale: { x: scaleX, y: scaleY }
    });

    // Create rectangle element
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    
    // Extract coordinates (assuming coords are [x1, y1, x2, y2])
    const [x1, y1, x2, y2] = coords;
    
    // Update SVG viewport to match original image dimensions for proper coordinate mapping
    overlay.setAttribute('viewBox', `0 0 ${imageNaturalWidth} ${imageNaturalHeight}`);
    overlay.style.width = `${imageDisplayWidth}px`;
    overlay.style.height = `${imageDisplayHeight}px`;
    
    // Set rectangle attributes using original image coordinates
    rect.setAttribute("x", Math.min(x1, x2));
    rect.setAttribute("y", Math.min(y1, y2));
    rect.setAttribute("width", Math.abs(x2 - x1));
    rect.setAttribute("height", Math.abs(y2 - y1));
    
    // Style the highlight
    rect.setAttribute("class", "segment-highlight");
    rect.setAttribute("fill", "rgba(255, 255, 0, 0.2)");  // Semi-transparent yellow
    rect.setAttribute("stroke", "rgba(255, 200, 0, 0.8)"); // More opaque yellow border
    rect.setAttribute("stroke-width", "2");
    
    // Add the highlight to the overlay
    overlay.appendChild(rect);

    console.log('[highlightSegment] Added highlight:', {
        coords,
        rect: {
            x: rect.getAttribute('x'),
            y: rect.getAttribute('y'),
            width: rect.getAttribute('width'),
            height: rect.getAttribute('height')
        }
    });
}

function toggleSegmentJson(show) {
    const container = document.getElementById('segment-json-display-container');
    const loadingEl = document.getElementById('segment-json-loading');
    const contentEl = document.getElementById('segment-json-content');
    
    if (!container || !loadingEl || !contentEl) {
        console.error('[toggleSegmentJson] Required elements not found');
        return;
    }
    
    if (show) {
        container.style.display = 'block';
        loadingEl.style.display = 'block';
        contentEl.style.display = 'none';
        fetchSegmentJson();
    } else {
        container.style.display = 'none';
        contentEl.textContent = '';
    }
}

function fetchSegmentJson() {
    const loadingEl = document.getElementById('segment-json-loading');
    const contentEl = document.getElementById('segment-json-content');
    
    if (!loadingEl || !contentEl) {
        console.error('[fetchSegmentJson] Required elements not found');
        return;
    }
    
    // Debug logging
    console.log('[fetchSegmentJson] Current state:', {
        index: currentSegmentDisplayIndex,
        listLength: currentSegmentsList?.length,
        folder: currentSegmentFolderGlobal,
        pageId: currentSegmentPageIdGlobal
    });

    // Validate segment index
    if (currentSegmentDisplayIndex < 0 || !currentSegmentsList || currentSegmentDisplayIndex >= currentSegmentsList.length) {
        const error = 'No segment is currently selected or invalid segment index.';
        console.error('[fetchSegmentJson]', error);
        contentEl.textContent = `Error: ${error}`;
        loadingEl.style.display = 'none';
        contentEl.style.display = 'block';
        return;
    }
    
    // Get current segment
    const segment = currentSegmentsList[currentSegmentDisplayIndex];
    
    // Validate segment object
    if (!segment) {
        const error = 'Selected segment data is null or undefined.';
        console.error('[fetchSegmentJson]', error);
        contentEl.textContent = `Error: ${error}`;
        loadingEl.style.display = 'none';
        contentEl.style.display = 'block';
        return;
    }

    // Get segment ID (handle both string and number types)
    const segmentId = segment.id !== undefined ? String(segment.id) : 
                     currentSegmentDisplayIndex.toString().padStart(3, '0');
    
    if (!segmentId) {
        const error = 'Unable to determine segment ID.';
        console.error('[fetchSegmentJson]', error);
        contentEl.textContent = `Error: ${error}`;
        loadingEl.style.display = 'none';
        contentEl.style.display = 'block';
        return;
    }

    // Get user ID from document
    const userId = document.querySelector('meta[name="user-id"]')?.content || 'anonymous';
    const folder = currentSegmentFolderGlobal;
    const pageId = currentSegmentPageIdGlobal;
    
    if (!userId || !folder || !pageId) {
        const error = 'Missing required information to fetch JSON data.';
        console.error('[fetchSegmentJson]', error, { userId, folder, pageId });
        contentEl.textContent = `Error: ${error}`;
        loadingEl.style.display = 'none';
        contentEl.style.display = 'block';
        return;
    }
    
    // Construct URL with proper encoding
    const jsonUrl = `/get_segment_json/${encodeURIComponent(userId)}/${encodeURIComponent(folder)}/${encodeURIComponent(pageId)}/${encodeURIComponent(segmentId)}`;
    
    console.log('[fetchSegmentJson] Fetching JSON from:', jsonUrl);
    
    // Show loading state
    loadingEl.style.display = 'block';
    contentEl.style.display = 'none';
    
    fetch(jsonUrl)
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server error ${response.status}: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Format JSON with indentation for readability
            contentEl.textContent = JSON.stringify(data, null, 2);
            loadingEl.style.display = 'none';
            contentEl.style.display = 'block';
        })
        .catch(error => {
            console.error('[fetchSegmentJson] Error:', error);
            contentEl.textContent = `Error loading JSON: ${error.message}`;
            loadingEl.style.display = 'none';
            contentEl.style.display = 'block';
        });
}

function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-segment-btn');
    const nextBtn = document.getElementById('next-segment-btn');
    if (!prevBtn || !nextBtn) return;

    prevBtn.disabled = (currentSegmentDisplayIndex <= 0);
    nextBtn.disabled = (currentSegmentDisplayIndex >= currentSegmentsList.length - 1);
    // UX: Add a visual cue for disabled state if not handled by CSS
    prevBtn.style.opacity = prevBtn.disabled ? 0.6 : 1;
    nextBtn.style.opacity = nextBtn.disabled ? 0.6 : 1;
}

function displaySegmentByIndex(index, updateUrl = true) {
    if (index >= 0 && index < currentSegmentsList.length) {
        currentSegmentDisplayIndex = index;
        const segment = currentSegmentsList[index];
        
        // Clear any existing highlight first
        clearSegmentHighlight();
        
        // If segment has coordinates, highlight them immediately
        if (segment.coords && Array.isArray(segment.coords) && segment.coords.length === 4) {
            console.log('[displaySegmentByIndex] Highlighting with coords from segment:', segment.coords);
            highlightSegment(segment.coords);
        }
        
        // Update navigation buttons state
        updateNavigationButtons();

        // Update segment buttons highlighting
        document.querySelectorAll('.line-segments-nav .segment-btn').forEach((btn, i) => {
            if (i === index) {
                btn.style.backgroundColor = '#e9ecef';
                btn.style.borderColor = '#adb5bd';
                btn.style.fontWeight = 'bold';
            } else {
                btn.style.backgroundColor = '#f1f3f5';
                btn.style.borderColor = '#dee2e6';
                btn.style.fontWeight = 'normal';
            }
        });

        // Update dropdown without triggering form submit
        const segmentSelect = document.getElementById('segment-select');
        if (segmentSelect) {
            segmentSelect.value = index;
        }

        // Update URL if requested
        if (updateUrl) {
            const url = new URL(window.location);
            url.searchParams.set('selected_segment', index);
            history.pushState({ segmentIndex: index }, '', url);
        }

        // Get user ID from document
        const userId = document.querySelector('meta[name="user-id"]')?.content || 'anonymous';
        
        // Fetch and display segment preview
        const previewUrl = `/get_segment_preview/${encodeURIComponent(userId)}/${encodeURIComponent(currentSegmentFolderGlobal)}/${encodeURIComponent(currentSegmentPageIdGlobal)}/${index}`;
        
        // Show loading state
        const imageContainer = document.querySelector('.selected-line-image');
        const textContainer = document.querySelector('.selected-line-text');
        if (imageContainer) imageContainer.innerHTML = '<p style="text-align:center; color:#6c757d;">Loading image...</p>';
        if (textContainer) textContainer.innerHTML = '<p style="text-align:center; color:#6c757d;">Loading text...</p>';

        fetch(previewUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                if (data.success && data.data) {
                    // Update image
                    if (imageContainer && data.data.imageUrl) {
                        imageContainer.innerHTML = `
                            <img src="${data.data.imageUrl}" 
                                 alt="Segment ${index + 1}" 
                                 style="max-width: 100%; max-height:150px; border: 1px solid #dee2e6; object-fit: contain; background-color: #fff;"
                                 onerror="this.onerror=null; this.src=''; this.parentElement.innerHTML='<p style=\\\'color:red;\\\'>Error loading image</p>';">
                        `;
                    }

                    // Update text
                    if (textContainer) {
                        let displayText = data.data.text || '(No text available)';
                        if (data.data.confidence) {
                            displayText += ` (Confidence: ${data.data.confidence.toFixed(3)})`;
                        }
                        textContainer.textContent = displayText;
                    }

                    // Update highlight with coordinates from the fetched data
                    if (data.data.coords && Array.isArray(data.data.coords) && data.data.coords.length === 4) {
                        console.log('[displaySegmentByIndex] Highlighting with coords from API:', data.data.coords);
                        highlightSegment(data.data.coords);
                    }
                } else {
                    throw new Error(data.error || 'Failed to load segment preview');
                }
            })
            .catch(error => {
                console.error('Error loading segment preview:', error);
                if (imageContainer) imageContainer.innerHTML = '<p style="color:red;">Error loading image</p>';
                if (textContainer) textContainer.innerHTML = '<p style="color:red;">Error loading text</p>';
                // Clear highlight on error
                clearSegmentHighlight();
            });
    } else {
        // Clear highlight if no valid segment is selected
        clearSegmentHighlight();
    }
}

function checkForExistingSegments(folder, pageId) {
    const segmentLoading = document.getElementById('segment-loading');
    const segmentSelect = document.getElementById('segment-select');
    
    if (!segmentLoading || !segmentSelect) return;
    
    segmentLoading.innerHTML = '<span class="loading-dots">Checking for existing segments</span>';
    
    while (segmentSelect.options.length > 1) {
        segmentSelect.remove(1);
    }
    
    // Get user ID from document
    const userId = document.querySelector('meta[name="user-id"]')?.content || 'anonymous';
    
    const segmentsUrl = `/get_segments/${encodeURIComponent(userId)}/${encodeURIComponent(folder)}/${encodeURIComponent(pageId)}`;
    
    console.log('[checkForExistingSegments] Fetching segments from:', segmentsUrl);
    
    fetch(segmentsUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load segments: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('[checkForExistingSegments] Segments response:', data);
            
            if (data.success && data.segments && Array.isArray(data.segments) && data.segments.length > 0) {
                data.segments.forEach((segment, index) => { // index is 0-based
                    const option = document.createElement('option');
                    option.value = index; // Use 0-based index for the option value

                    let displayText = `Segment ${index + 1}`; // Default display text
                    if (segment.text) {
                        let textSample = '';
                        if (typeof segment.text === 'string') {
                            textSample = segment.text;
                        } else if (Array.isArray(segment.text) && typeof segment.text[0] === 'string') {
                            textSample = segment.text[0];
                        }
                        if (textSample) {
                            const truncatedText = textSample.substring(0, 30);
                            displayText = `Segment ${index + 1}: ${truncatedText}${textSample.length > 30 ? '...' : ''}`;
                        }
                    } else if (segment.id) { // Fallback to segment.id if no text
                        displayText = `Segment ${index + 1} (ID: ${segment.id})`;
                    }
                    
                    option.textContent = displayText;
                    segmentSelect.appendChild(option);
                });
                
                segmentLoading.textContent = `${data.segments.length} segments available.`;
                
                const urlParams = new URLSearchParams(window.location.search);
                const selectedSegmentFromUrl = urlParams.get('selected_segment'); // This should be an index
                
                if (selectedSegmentFromUrl !== null) { // Check for null or undefined explicitly
                    segmentSelect.value = selectedSegmentFromUrl;
                }

                // Store segments globally for reference
                currentSegmentsList = data.segments;
                currentSegmentFolderGlobal = folder;
                currentSegmentPageIdGlobal = pageId;

                // If sidebar dropdown is populated, try to load segments into the main navigation area
                console.log('[checkForExistingSegments] Sidebar dropdown populated, calling loadLineSegments.');
                loadLineSegments(folder, pageId);

            } else {
                segmentLoading.textContent = "No segments found. Run OCR to detect segments.";
                if (data.error) {
                    console.warn(`[checkForExistingSegments] Error from /get_segments: ${data.error}`);
                }
                // Ensure the main segment navigation area is hidden if no segments for dropdown
                const lineSegmentsContainer = document.getElementById('line-segments-container');
                if (lineSegmentsContainer) {
                    lineSegmentsContainer.style.display = 'none';
                }
            }
        })
        .catch(error => {
            console.error('[checkForExistingSegments] Error loading segments:', error);
            segmentLoading.innerHTML = '<span style="color:red;">Error loading segments.</span> Run OCR to detect segments.';
            // Ensure the main segment navigation area is hidden on error
            const lineSegmentsContainer = document.getElementById('line-segments-container');
            if (lineSegmentsContainer) {
                lineSegmentsContainer.style.display = 'none';
            }
        });
}

function populateSegmentsDropdown(folder, pageId, segments) {
    const segmentSelect = document.getElementById('segment-select');
    const segmentLoading = document.getElementById('segment-loading');
    if (!segmentSelect) return;
    
    console.log('[populateSegmentsDropdown] Populating with segments:', segments);
    
    while (segmentSelect.options.length > 1) {
        segmentSelect.remove(1);
    }
    
    if (segments && Array.isArray(segments) && segments.length > 0) {
        segments.forEach((segment, index) => { // index is 0-based
            const option = document.createElement('option');
            option.value = index; // Use 0-based index for the option value

            let displayText = `Segment ${index + 1}`; // Default display text
            if (segment.text) {
                let textSample = '';
                if (typeof segment.text === 'string') {
                    textSample = segment.text;
                } else if (Array.isArray(segment.text) && typeof segment.text[0] === 'string') {
                    textSample = segment.text[0];
                } else if (segment.status) { // From OCR service direct status
                    textSample = `Status: ${segment.status}`;
                } else if (segment.file) { // From OCR service direct file
                     textSample = `File: ${segment.file.split('/').pop()}`;
                }
                if (textSample) {
                    const truncatedText = textSample.substring(0, 30);
                    displayText = `Segment ${index + 1}: ${truncatedText}${textSample.length > 30 ? '...' : ''}`;
                }
            } else if (segment.id) { // Fallback to segment.id if no text
                 displayText = `Segment ${index + 1} (ID: ${segment.id})`;
            }
            
            option.textContent = displayText;
            segmentSelect.appendChild(option);
        });
        if (segmentLoading) segmentLoading.textContent = `${segments.length} segments detected.`;

        const urlParams = new URLSearchParams(window.location.search);
        const selectedSegmentFromUrl = urlParams.get('selected_segment'); // This should be an index
        
        if (selectedSegmentFromUrl !== null) {
            segmentSelect.value = selectedSegmentFromUrl;
        }

        // Store segments globally for reference
        currentSegmentsList = segments;
        currentSegmentFolderGlobal = folder;
        currentSegmentPageIdGlobal = pageId;

    } else {
         if (segmentLoading) segmentLoading.textContent = "No segments detected by OCR.";
    }
}

function performOCR() {
    const resultArea = document.getElementById('ocr-result');
    const ocrButton = document.querySelector('.ocr-section button');
    const lineSegmentsContainer = document.getElementById('line-segments-container');
    const segmentLoading = document.getElementById('segment-loading');

    ocrButton.disabled = true;
    ocrButton.textContent = "Processing...";
    ocrButton.style.opacity = 0.6; // UX: Visual cue for disabled
    resultArea.value = "Initializing text recognition..."; // UX: Initial message
    if(lineSegmentsContainer) lineSegmentsContainer.style.display = 'none';
    if (segmentLoading) segmentLoading.innerHTML = '<span class="loading-dots">Processing segments</span>';

    // Get parameters from URL
    const urlParams = new URLSearchParams(window.location.search);
    const selectedPng = urlParams.get('selected_png');
    const selectedFolder = urlParams.get('selected_folder');

    fetch('/process_ocr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: selectedFolder, filename: selectedPng })
    })
    .then(response => {
        resultArea.value = "Analyzing page content, please wait...";
        if (!response.ok) {
            throw new Error(`OCR server returned status ${response.status}`);
        }
        return response.json();
    })
    .then((data) => {
        console.log('performOCR - OCR Response:', data);
        if (data.success) {
            // Safely set the main OCR text result
            if (typeof data.text === 'string') {
                resultArea.value = data.text.trim() || "(No text content recognized)";
            } else if (data.segments && data.segments.length > 0) {
                let fullText = data.segments.map(s => {
                    if (typeof s.text === 'string') return s.text;
                    if (Array.isArray(s.text) && typeof s.text[0] === 'string') return s.text[0];
                    return '';
                }).join('\n');
                resultArea.value = fullText.trim() || "Segments processed. See details below or select a segment.";
            } else if (data.status) {
                 resultArea.value = `OCR Status: ${data.status}${data.file ? '\nFile: ' + data.file : ''}`;
            } else {
                resultArea.value = "OCR processing complete. No text content found or segments detected.";
            }

            // Update the sidebar segments dropdown
            if (data.page_id && data.segments && data.segments.length > 0) {
                populateSegmentsDropdown(selectedFolder, data.page_id, data.segments);
                if (segmentLoading) segmentLoading.textContent = `${data.segments.length} segments detected.`;
            } else {
                if (segmentLoading) segmentLoading.textContent = "No segments detected by OCR.";
            }
            
            // Load the main content line segment navigation buttons
            if (data.page_id) {
                console.log('performOCR - Calling loadLineSegments for page_id:', data.page_id);
                loadLineSegments(selectedFolder, data.page_id);
            } else {
                console.warn('performOCR - No page_id in OCR response, cannot load line segments.');
                 if(lineSegmentsContainer) lineSegmentsContainer.style.display = 'none';
            }
        } else {
            resultArea.value = "Error performing OCR: " + (data.error || "Unknown error. Please check console.");
            if (segmentLoading) segmentLoading.innerHTML = '<span style="color:red;">OCR failed.</span> Please try again.';
            console.error('performOCR - OCR Error:', data.error);
        }
    })
    .catch(error => {
        resultArea.value = "Error performing OCR: " + error.message + " (Check network connection and console).";
        console.error('performOCR - Fetch/Network Error:', error);
        if (segmentLoading) segmentLoading.innerHTML = '<span style="color:red;">OCR error.</span> Please try again.';
    })
    .finally(() => {
        ocrButton.disabled = false;
        ocrButton.textContent = "Extract Text from Page";
        ocrButton.style.opacity = 1; // UX: Reset opacity
    });
}

function loadLineSegments(folder, pageId) {
    const lineSegmentsContainer = document.getElementById('line-segments-container');
    const lineSegmentsNav = document.querySelector('.line-segments-nav');
    const selectedLineImageEl = document.querySelector('.selected-line-image');
    const selectedLineTextEl = document.querySelector('.selected-line-text');
    
    console.log(`[loadLineSegments] Called for folder: "${folder}", pageId: "${pageId}"`);

    if (!lineSegmentsContainer || !lineSegmentsNav || !selectedLineImageEl || !selectedLineTextEl) {
        console.error("[loadLineSegments] Critical HTML elements for segment navigation not found!");
        return;
    }

    lineSegmentsNav.innerHTML = '<p style="font-size:0.9em; color:#6c757d; padding:10px;" class="loading-dots">Loading line segments</p>';
    selectedLineImageEl.innerHTML = ''; 
    selectedLineTextEl.innerText = ''; 
    lineSegmentsContainer.style.display = 'block'; // UX: Show container to display loading/message
    
    // Get user ID from document
    const userId = document.querySelector('meta[name="user-id"]')?.content || 'anonymous';
    
    // Reset global variables
    currentSegmentFolderGlobal = folder;
    currentSegmentPageIdGlobal = pageId;
    currentSegmentDisplayIndex = -1;
    updateNavigationButtons(); // Initially disable buttons

    const segmentsUrl = `/get_segments/${encodeURIComponent(userId)}/${encodeURIComponent(folder)}/${encodeURIComponent(pageId)}`;
    
    console.log('[loadLineSegments] Fetching segments from:', segmentsUrl);

    fetch(segmentsUrl)
        .then(response => {
            console.log(`[loadLineSegments] Fetch response status for "${segmentsUrl}": ${response.status}`);
            if (!response.ok) {
                // Try to get error message from response body if possible
                return response.text().then(text => {
                    throw new Error(`Failed to load segments for navigation: ${response.status}. Server response: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('[loadLineSegments] Segments data received:', data);
            lineSegmentsNav.innerHTML = ''; // Clear loading message

            if (data && data.success === true && data.segments && Array.isArray(data.segments) && data.segments.length > 0) {
                currentSegmentsList = data.segments; // Store fetched segments globally
                console.log(`[loadLineSegments] ${currentSegmentsList.length} segments found. Displaying container.`);

                currentSegmentsList.forEach((segment, index) => {
                    if (!segment || typeof segment !== 'object') {
                        console.warn(`[loadLineSegments] Segment at index ${index} is not a valid object:`, segment);
                        return; 
                    }

                    const segmentBtn = document.createElement('button');
                    segmentBtn.className = 'segment-btn';
                    segmentBtn.style.cssText = `
                        padding: 5px 10px; background-color: #f1f3f5; border: 1px solid #dee2e6;
                        border-radius: 4px; cursor: pointer; min-width: 100px;
                        text-align: center; font-size: 0.9em; margin: 2px; transition: background-color 0.2s, border-color 0.2s;`;
                    
                    let btnText = `L${index + 1}`;
                    let segmentFullText = `Segment ${index + 1}`;
                    if (segment.text) {
                        let textSample = '';
                        if (typeof segment.text === 'string') {
                            textSample = segment.text;
                        } else if (Array.isArray(segment.text) && typeof segment.text[0] === 'string') {
                            textSample = segment.text[0];
                        }
                        
                        if (textSample) {
                            segmentFullText = textSample; 
                            let shortSample = textSample.trim().substring(0,15);
                            btnText = `${index + 1}: ${shortSample}${textSample.length > 15 ? '...' : ''}`;
                        }
                    }
                    segmentBtn.textContent = btnText;
                    segmentBtn.title = segmentFullText; // UX: Tooltip for full text
                    segmentBtn.dataset.segmentIndex = index; // Store index for easy access

                    segmentBtn.onclick = function(e) {
                        e.preventDefault();
                        const clickedIndex = parseInt(this.dataset.segmentIndex, 10);
                        displaySegmentByIndex(clickedIndex);
                    };
                    lineSegmentsNav.appendChild(segmentBtn);
                });

                if (currentSegmentsList.length > 0) {
                    // Determine which segment to display initially
                    const urlParams = new URLSearchParams(window.location.search);
                    const segmentIndexFromUrl = urlParams.get('selected_segment');
                    let initialIndexToDisplay = 0; // Default to the first segment

                    if (segmentIndexFromUrl !== null) {
                        const parsedIndex = parseInt(segmentIndexFromUrl, 10);
                        if (!isNaN(parsedIndex) && parsedIndex >= 0 && parsedIndex < currentSegmentsList.length) {
                            initialIndexToDisplay = parsedIndex;
                            console.log(`[loadLineSegments] Initial segment index from URL: ${initialIndexToDisplay}`);
                        } else {
                            console.warn(`[loadLineSegments] Invalid segment index from URL: ${segmentIndexFromUrl}. Defaulting to 0.`);
                        }
                    }
                    displaySegmentByIndex(initialIndexToDisplay); // Display the segment from URL or the first one
                } else {
                     updateNavigationButtons(); 
                     lineSegmentsNav.innerHTML = '<p style="font-size:0.9em; color:#6c757d; padding:10px;">No line segments found.</p>';
                }
            } else {
                console.log('[loadLineSegments] No valid segments found.');
                lineSegmentsNav.innerHTML = '<p style="font-size:0.9em; color:#6c757d; padding:10px;">No line segments were detected for this page.</p>';
                selectedLineImageEl.innerHTML = '';
                selectedLineTextEl.innerText = '';
                currentSegmentsList = [];
                currentSegmentDisplayIndex = -1;
                updateNavigationButtons();
                if (data && data.error) console.warn(`[loadLineSegments] Error from /get_segments: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('[loadLineSegments] Fetch/Processing Error for segments:', error);
            lineSegmentsNav.innerHTML = `<p style="font-size:0.9em; color:red; padding:10px;">Error loading line segments: ${error.message}</p>`;
            selectedLineImageEl.innerHTML = '';
            selectedLineTextEl.innerText = '';
            currentSegmentsList = [];
            currentSegmentDisplayIndex = -1;
            updateNavigationButtons();
        });
}