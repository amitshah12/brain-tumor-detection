/******NEW code 1*****/
import { useState, useEffect } from "react";
import '../App.css';
import axios from "axios";

const Imageinput = ({ setPrediction }) => {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            // Clean up previous preview URL
            if (preview) {
                URL.revokeObjectURL(preview);
            }
            // Create new preview URL
            const previewUrl = URL.createObjectURL(file);
            setPreview(previewUrl);
        } else {
            setImage(null);
            setPreview(null);
        }
    };

    const handleClick = async (e) => {
        e.preventDefault();
        
        if (!image) {
            alert("Please select an image first");
            return;
        }

        setLoading(true);
        
        const api_uri = import.meta.env.VITE_REST_API || 'http://localhost:5000';
        const form = new FormData();
        form.append("image", image);

        try {
            const result = await axios.post(`${api_uri}/predict`, form, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            
            setPrediction(result.data);
            
        } catch (error) {
            console.error("Error uploading image:", error);
            alert("Error processing image. Please try again.");
            setPrediction({
                error: "Error processing image. Please try again."
            });
        } finally {
            setLoading(false);
        }
    };

    // Clean up preview URL when component unmounts
    useEffect(() => {
        return () => {
            if (preview) {
                URL.revokeObjectURL(preview);
            }
        };
    }, [preview]);

    return (
        <form className="image-input-container">
            <div className="input-field">
                <input 
                    type="file" 
                    accept="image/*"
                    onChange={handleImageChange}
                    disabled={loading}
                />
                <button 
                    type="button"
                    onClick={handleClick} 
                    disabled={!image || loading}
                >
                    {loading ? 'Processing...' : 'Predict'}
                </button>
            </div>
            
            {preview && (
                <div className="preview-container">
                    <h3>Selected Image:</h3>
                    <img 
                        src={preview} 
                        alt="MRI Preview" 
                        className="image-preview"
                    />
                </div>
            )}
        </form>
    );
};

export default Imageinput;
