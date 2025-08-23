/******NEW code 1*****/
// import { useState, useEffect } from "react";
// import '../App.css';
// import axios from "axios";

// const Imageinput = ({ setPrediction }) => {
//     const [image, setImage] = useState(null);
//     const [preview, setPreview] = useState(null);
//     const [loading, setLoading] = useState(false);

//     const handleImageChange = (e) => {
//         const file = e.target.files[0];
//         if (file) {
//             setImage(file);
//             // Clean up previous preview URL
//             if (preview) {
//                 URL.revokeObjectURL(preview);
//             }
//             // Create new preview URL
//             const previewUrl = URL.createObjectURL(file);
//             setPreview(previewUrl);
//         } else {
//             setImage(null);
//             setPreview(null);
//         }
//     };

//     const handleClick = async (e) => {
//         e.preventDefault();
        
//         if (!image) {
//             alert("Please select an image first");
//             return;
//         }

//         setLoading(true);
        
//         const api_uri = import.meta.env.VITE_REST_API || 'http://localhost:5000';
//         const form = new FormData();
//         form.append("image", image);

//         try {
//             const result = await axios.post(`${api_uri}/predict`, form, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data'
//                 }
//             });
            
//             setPrediction(result.data);
            
//         } catch (error) {
//             console.error("Error uploading image:", error);
//             alert("Error processing image. Please try again.");
//             setPrediction({
//                 error: "Error processing image. Please try again."
//             });
//         } finally {
//             setLoading(false);
//         }
//     };

//     // Clean up preview URL when component unmounts
//     useEffect(() => {
//         return () => {
//             if (preview) {
//                 URL.revokeObjectURL(preview);
//             }
//         };
//     }, [preview]);

//     return (
//         <form className="image-input-container">
//             <div className="input-field">
//                 <input 
//                     type="file" 
//                     accept="image/*"
//                     onChange={handleImageChange}
//                     disabled={loading}
//                 />
//                 <button 
//                     type="button"
//                     onClick={handleClick} 
//                     disabled={!image || loading}
//                 >
//                     {loading ? 'Processing...' : 'Predict'}
//                 </button>
//             </div>
            
//             {preview && (
//                 <div className="preview-container">
//                     <h3>Selected Image:</h3>
//                     <img 
//                         src={preview} 
//                         alt="MRI Preview" 
//                         className="image-preview"
//                     />
//                 </div>
//             )}
//         </form>
//     );
// };

// export default Imageinput;


/********AFTER ADDING DRIVE LINK FOR MODEL.H5*********/
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
            if (preview) {
                URL.revokeObjectURL(preview);
            }
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
        
        // Use production API URL or fallback to development
        const api_uri = import.meta.env.VITE_REST_API || 'http://localhost:5000';
        const form = new FormData();
        form.append("image", image);

        try {
            console.log("Uploading to:", `${api_uri}/predict`);
            
            const result = await axios.post(`${api_uri}/predict`, form, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                timeout: 120000 // 2 minutes timeout for production (model download on first request)
            });
            
            console.log("Prediction result:", result.data);
            setPrediction(result.data);
            
        } catch (error) {
            console.error("Error uploading image:", error);
            
            let errorMessage = "Error processing image. ";
            if (error.response) {
                if (error.response.status === 503) {
                    errorMessage += "Model is loading, please wait and try again in a few minutes.";
                } else {
                    errorMessage += `Server error: ${error.response.data?.error || error.response.status}`;
                }
            } else if (error.code === 'ECONNABORTED') {
                errorMessage += "Request timeout. The server might be starting up, please try again.";
            } else if (error.request) {
                errorMessage += "Cannot connect to server. Please check your connection.";
            } else {
                errorMessage += error.message;
            }
            
            alert(errorMessage);
            setPrediction({ error: errorMessage });
            
        } finally {
            setLoading(false);
        }
    };

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
