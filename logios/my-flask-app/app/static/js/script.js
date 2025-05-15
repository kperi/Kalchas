// This file contains JavaScript for client-side functionality, including image preview and cropping.

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('image-preview');
    const cropButton = document.getElementById('crop-button');
    let cropper;

    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewContainer.innerHTML = `<img id="image" src="${e.target.result}" alt="Image Preview" />`;
                const image = document.getElementById('image');
                cropper = new Cropper(image, {
                    aspectRatio: 16 / 9,
                    viewMode: 1,
                });
            };
            reader.readAsDataURL(file);
        }
    });

    cropButton.addEventListener('click', function() {
        if (cropper) {
            const canvas = cropper.getCroppedCanvas();
            canvas.toBlob(function(blob) {
                const formData = new FormData();
                formData.append('croppedImage', blob);
                // Send the cropped image to the server
                fetch('/upload_cropped_image', {
                    method: 'POST',
                    body: formData,
                }).then(response => {
                    if (response.ok) {
                        alert('Cropped image uploaded successfully!');
                    } else {
                        alert('Failed to upload cropped image.');
                    }
                });
            });
        }
    });
});