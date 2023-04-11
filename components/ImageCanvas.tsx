import { useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { inference } from '../utils/predict';
import styles from '../styles/Home.module.css';

interface Props {
  // eslint-disable-next-line no-unused-vars
  onImageLoad: (src: string) => void;
  // eslint-disable-next-line no-unused-vars
  onLabelChange: (label: []) => void; // Add this prop
}
let counter = 0;
const ImageCanvas = (props: Props) => {

  let image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");

  // Load the image from the IMAGE_URLS array
  const getImage = () => {
    const sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
    counter ++;
    if(counter > 2) {counter = 0;}
    return sampleImageUrls[counter];
  }

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = () => { 
    // Get the image
    image = new Image();
    const sampleImage = getImage();
    image.src = sampleImage.value;
    image.onload = () => {
      props.onImageLoad(image.src); // Notify the parent component about the loaded image
    }
    // Run the inference
    submitInference();
  };

  const submitInference = async () => {

    // Get the image data from the canvas and submit inference.
    const [inferenceResult, inferenceTime] = await inference(image.src);
    // Update the label and confidence
    setLabel(inferenceResult.toString());
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);
    props.onLabelChange(inferenceResult); // Call the onLabelChange prop
  };

  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Calculate Spherical Harmonic Coefficients
      </button>
      <br/>
      <span>{topResultLabel}</span>
      <span>{inferenceTime}</span>
    </>
  )

};

export default ImageCanvas;
