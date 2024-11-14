import React, { useState } from 'react';

const ImageGenerator = () => {
  const [textInput, setTextInput] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);

  const generateImage = async () => {
    setLoading(true);
    setImageUrl(''); // Clear previous image
    const response = await fetch(`https://image.pollinations.ai/prompt/${encodeURIComponent(textInput)}`);
    
    if (response.ok) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageUrl(url);
    } else {
      console.error('Error generating image');
    }
    setLoading(false);
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result); // Set the uploaded image as a URL
      };
      reader.readAsDataURL(file); // Read the file as data URL
    }
  };

  return (
    <div 
      style={{
        backgroundColor: '#ADD8E6',  // Light blue background
        minHeight: '100vh',          // Full screen height
        display: 'flex',             // Flexbox to center the content
        justifyContent: 'center',    // Center horizontally
        alignItems: 'center',        // Center vertically
      }}
    >
      <div
        style={{
          backgroundColor: 'white',   // White background for the rectangle box
          padding: '70px',            // Add padding inside the box
          borderRadius: '10px',       // Rounded corners for the box
          width: '90%',               // Make the box take up 90% of the width
          maxWidth: '600px',          // Limit max width to 600px
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', // Light shadow for the box
        }}
      >
        <h1 style={{ textAlign: 'center', color: 'orange' }}>Design Your Product</h1>

        {/* Text Input */}
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Describe your design (e.g., Red shirt with striped pattern)"
          rows="5"
          style={{
            width: '100%',
            padding: '10px',
            fontSize: '16px',
            borderRadius: '8px',
            marginBottom: '20px',
            border: '1px solid #ccc', // Border for the textbox
            resize: 'none',            // Disable resizing the textbox
          }}
        />
        
        {/* Upload Image Button */}
        <p style={{ fontSize: '16px', marginBottom: '10px' }}>
          Upload your own design (optional):
        </p>
        <div style={{ textAlign: 'left', marginBottom: '40px' }}>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{
              padding: '10px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              width: '35%',
              fontSize: '16px',
              cursor: 'pointer',
            }}
          />
        </div>

        {/* Generate Image Button */}
        <div style={{ textAlign: 'center' }}>
          <button 
            onClick={generateImage} 
            disabled={loading}
            style={{
              backgroundColor: '#FF8C00',  // Orange button
              color: 'dark grey',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '8px',
              fontSize: '18px',
              cursor: 'pointer',
              width: '60%',  // Make the button full width of the container
              marginBottom: '20px',
            }}
          >
            {loading ? 'Generating...' : 'Generate Image'}
          </button>
        </div>
        
        {/* Display Uploaded Image */}
        {uploadedImage && (
          <div style={{ textAlign: 'center' }}>
            <img
              src={uploadedImage}
              alt="Uploaded"
              style={{
                width: '100%',
                maxHeight: '300px',  // Limit the size of the uploaded image
                objectFit: 'contain',
                borderRadius: '8px',
                marginBottom: '20px',
              }}
            />
          </div>
        )}

        {/* Display Generated Image */}
        {imageUrl && (
          <div style={{ textAlign: 'center' }}>
            <img 
              src={imageUrl} 
              alt="Generated" 
              style={{
                width: '100%',
                maxHeight: '400px',  // Optional, set a max height for the image
                objectFit: 'contain', // Ensures image aspect ratio is maintained
                borderRadius: '8px',  // Rounded corners for the image
                marginBottom: '20px', // Space below the image
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageGenerator;


