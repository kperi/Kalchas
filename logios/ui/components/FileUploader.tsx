import React, { useState } from 'react';

const FileUploader = () => {
    const [file, setFile] = useState(null);
    const [croppedFile, setCroppedFile] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile && selectedFile.type === 'application/pdf') {
            setFile(selectedFile);
            setError('');
        } else {
            setError('Please upload a valid PDF file.');
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('No file selected for upload.');
            return;
        }

        // Implement the upload logic here
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                setCroppedFile(result.croppedFile); // Assuming the API returns the cropped file
                setFile(null); // Reset file input
            } else {
                setError('Upload failed. Please try again.');
            }
        } catch (error) {
            setError('An error occurred during upload.');
        }
    };

    return (
        <div>
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            {error && <p style={{ color: 'red' }}>{error}</p>}
            <button onClick={handleUpload}>Upload</button>
            {croppedFile && <p>File uploaded successfully: {croppedFile.name}</p>}
        </div>
    );
};

export default FileUploader;