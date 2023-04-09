import { useRef, useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { inferenceSqueezenet } from '../utils/predict';
import styles from '../styles/Home.module.css';

interface Props {
  height: number;
  width: number;
}

const ImageCanvas = (props: Props) => {

  const canvasRef = useRef<HTMLCanvasElement>(null);
  let image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");
  
  // Load the image from the IMAGE_URLS array
  const getImage = () => {
    const sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
    return sampleImageUrls[0];
  }

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = () => { 
    // Get the image
    image = new Image();
    const sampleImage = getImage();
    image.src = sampleImage.value;

    // Clear out previous values.
    setLabel(`Inferencing...`);
    setInferenceTime("");

    // Draw the image on the canvas
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    }
   
    // Run the inference
    submitInference();
  };

  const submitInference = async () => {

    // Get the image data from the canvas and submit inference.
    const [inferenceResult, inferenceTime] = await inferenceSqueezenet(image.src);
    console.log(inferenceResult);
    // Update the label and confidence
    setLabel(inferenceResult.toString());
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

  };

  return (
    <>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run Squeezenet inference
      </button>
      <br/>
      <canvas ref={canvasRef} width={props.width} height={props.height} />
      <span>{topResultLabel}</span>
      <span>{inferenceTime}</span>
    </>
  )

};

export default ImageCanvas;
