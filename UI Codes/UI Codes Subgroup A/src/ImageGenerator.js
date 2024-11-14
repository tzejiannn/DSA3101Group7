import React, { useState } from 'react';
import axios from 'axios';

const ImageGenerator = () => {
  // State for the inputs and results
  const [type, setType] = useState('');
  const [color, setColor] = useState('');
  const [design, setDesign] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);  // State to handle uploaded image

  // Recommendations state
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState('');

  // Rating state
  const [rating, setRating] = useState(null); // Rating (1-5 stars)
  const [isRatingSubmitted, setIsRatingSubmitted] = useState(false); // Track if rating has been submitted

  // Function to combine inputs and generate the image
  const generateImage = async () => {
    // Ensure all fields are filled before sending the request
    if (!type || !color || !design) {
      alert('Please fill in all fields!');
      return;
    }

    // Combine the inputs into a single prompt
    const prompt = `${type} ${color} ${design}`;

    setLoading(true);
    setError('');

    try {
      // Send the combined prompt to the Pollinations API
      const response = await axios.get('https://image.pollinations.ai/prompt/' + encodeURIComponent(prompt), {
        params: {
          model: 'default',  // Optional, you can change this
          width: 1024,       // Increase width to 1024px
          height: 1024,      // Increase height to 1024px
          nologo: false,
          private: false,
          enhance: false,
        },
        responseType: 'arraybuffer',  // We're expecting an image buffer
      });

      // Convert the response data (image) into a Blob and create a URL for it
      const imageBlob = new Blob([response.data], { type: 'image/png' });
      const imageObjectURL = URL.createObjectURL(imageBlob);

      // Set the image URL to display it
      setImageUrl(imageObjectURL);
    } catch (err) {
      setError('Error generating image.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Function to handle the file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Create a URL for the uploaded image
      const fileUrl = URL.createObjectURL(file);
      setUploadedImage(fileUrl);  // Set the uploaded image URL
    }
  };

  // Function to get product recommendations
  const getRecommendations = async () => {
    setRecommendationLoading(true);
    setRecommendationError('');

    const customerData = {
      apparel_type: type,
      color: color,
      design: design
    };

    try {
      const response = await fetch('http://127.0.0.1:5000/recommendations/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(customerData),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      if (Array.isArray(data) && data.length > 0) {
        setRecommendations(data); // Set recommendations to the response data
      } else {
        throw new Error('No recommendations found');
      }
    } catch (error) {
      setRecommendationError('Failed to fetch product recommendations');
      console.error(error);
    } finally {
      setRecommendationLoading(false);
    }
  };

  // Function to handle rating submission
  const handleRatingSubmit = () => {
    if (rating === null) {
      alert('Please select a rating!');
    } else {
      // Handle the submission (you can send the rating to a backend here)
      console.log('Rating submitted:', rating);
      setIsRatingSubmitted(true); // Set flag to true after submitting rating
    }
  };

  return (
    <div style={{ backgroundColor: '#add8e6', minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ width: '80%', maxWidth: '900px', padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 4px 10px rgba(0, 0, 0, 0.1)', display: 'flex', flexDirection: 'row', gap: '20px' }}>
        
        {/* Left Section - Form to input data */}
        <div style={{ flex: 1 }}>
          <h1 style={{ textAlign: 'center', color: 'orange' }}>Design Your Product</h1>

          {/* Color */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '16px', fontWeight: 'bold' }}>
              Color:
            </label>
            <input
              type="text"
              placeholder="Enter color (e.g., red, blue, green)"
              value={color}
              onChange={(e) => setColor(e.target.value)}
              style={{ width: '90%', padding: '10px', fontSize: '16px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>

          {/* Type of Apparel */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '16px', fontWeight: 'bold' }}>
              Type of Apparel:
            </label>
            <input
              type="text"
              placeholder="Enter type of apparel (e.g., shirt, pants, jacket)"
              value={type}
              onChange={(e) => setType(e.target.value)}
              style={{ width: '90%', padding: '10px', fontSize: '16px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>

          {/* Design (Larger Textbox with Placeholder) */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '16px', fontWeight: 'bold' }}>
              Design:
            </label>
            <textarea
              placeholder="Enter your description (e.g., with red dots)"
              value={design}
              onChange={(e) => setDesign(e.target.value)}
              rows="6"  // Makes the textbox taller (6 rows)
              style={{
                width: '90%',  // Full width
                padding: '10px',
                fontSize: '14px',
                borderRadius: '4px',
                border: '1px solid #ccc',
                resize: 'none',  // Prevent resizing
              }}
            />
          </div>

          {/* Product Recommendations */}
          <div style={{ marginBottom: '20px' }}>
            <button 
              onClick={getRecommendations} 
              disabled={recommendationLoading}
              style={{
                padding: '10px 20px', 
                backgroundColor: 'orange', 
                color: 'dark grey', 
                border: 'none', 
                borderRadius: '4px', 
                fontSize: '16px', 
                cursor: 'pointer'
              }}>
              {recommendationLoading ? 'Loading Recommendations...' : 'Get Product Recommendations'}
            </button>

            {recommendationError && <p style={{ color: 'red' }}>{recommendationError}</p>}

            {/* Display recommendations */}
            {recommendations.length > 0 && (
              <div>
                <h3>Recommended Products:</h3>
                <ul>
                  {recommendations.map((item, index) => (
                    <li key={index}>{item.apparel_type} - {item.color} - {item.design}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Upload Image Button */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '16px', fontWeight: 'bold' }}>
              Upload Your Own Design (Optional):
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              style={{
                display: 'none', // Hide the default input box
              }}
              id="upload-image"
            />
            <label 
              htmlFor="upload-image" 
              style={{
                display: 'inline-block',
                width: '60px',
                height: '60px',
                backgroundColor: '#4CAF50',
                borderRadius: '8px',
                textAlign: 'center',
                lineHeight: '60px',
                color: 'white',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '14px',
                marginTop: '10px'
              }}
            >
              +
            </label>
          </div>

          {/* Button to Generate Image */}
          <div style={{ textAlign: 'center'}}>
            <button 
              onClick={generateImage} 
              disabled={loading}
              style={{ 
                padding: '10px 20px', 
                backgroundColor: 'orange', 
                color: 'dark grey', 
                border: 'none', 
                borderRadius: '4px', 
                fontSize: '16px', 
                cursor: 'pointer' 
              }}>
              {loading ? 'Generating...' : 'Generate Image'}
            </button>
          </div>

          {/* Submit Button to Show Rating Prompt */}
          {imageUrl && !isRatingSubmitted && (
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <button
                onClick={() => setIsRatingSubmitted(true)}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'green',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '16px',
                  cursor: 'pointer'
                }}
              >
                Submit Image
              </button>
            </div>
          )}

          {/* Rating Prompt (appears after submitting the image) */}
          {isRatingSubmitted && (
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <h3>How satisfied with the image Generated</h3>
              <div>
                {[1, 2, 3, 4, 5].map((star) => (
                  <span 
                    key={star} 
                    style={{
                      fontSize: '24px', 
                      cursor: 'pointer', 
                      color: star <= rating ? 'gold' : 'gray'
                    }}
                    onClick={() => setRating(star)}
                  >
                    ★
                  </span>
                ))}
              </div>
              <button
                onClick={handleRatingSubmit}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'orange',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '16px',
                  cursor: 'pointer',
                  marginTop: '10px'
                }}
              >
                Submit Rating
              </button>
            </div>
          )}

          {/* Error message if image generation fails */}
          {error && <p style={{ color: 'red', textAlign: 'center' }}>{error}</p>}
        </div>

        {/* Right Section - Display the Generated Image */}
        <div style={{
          flex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          padding: '20px',
          border: '2px solid #ccc',  // Outline for the image area
          borderRadius: '8px',
          minHeight: '400px',  // Minimum height to show the outline
          width: 'calc(100% - 2cm)',  // Reduce width by 1 cm on each side
          backgroundColor: imageUrl || uploadedImage ? 'transparent' : '#f0f0f0',  // Light background when no image
        }}>
          {uploadedImage ? (
            <img 
              src={uploadedImage} 
              alt="Uploaded"
              style={{ 
                width: '100%',  
                height: '100%', 
                objectFit: 'cover', 
                borderRadius: '8px' 
              }} 
            />
          ) : imageUrl ? (
            <img 
              src={imageUrl} 
              alt="Generated" 
              style={{ 
                width: '100%', 
                height: '100%', 
                objectFit: 'cover', 
                borderRadius: '8px' 
              }} 
            />
          ) : (
            <p style={{ fontSize: '18px', color: '#888' }}>
              Image
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageGenerator;
