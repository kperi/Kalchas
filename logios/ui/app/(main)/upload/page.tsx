import React, { useState } from 'react';
import FileUploader from '../../../components/FileUploader';

const UploadPage = () => {
    const [uploadStatus, setUploadStatus] = useState('');

    const handleUploadSuccess = () => {
        setUploadStatus('File uploaded successfully!');
    };

    const handleUploadError = () => {
        setUploadStatus('Error uploading file. Please try again.');
    };

    return (
        <div>
            <h1>Upload PDF</h1>
            <FileUploader 
                onSuccess={handleUploadSuccess} 
                onError={handleUploadError} 
            />
            {uploadStatus && <p>{uploadStatus}</p>}
        </div>
    );
};

export default UploadPage;