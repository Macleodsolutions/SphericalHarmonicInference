import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export async function loadSHCModel(): Promise<void> {
  try {
    // Create an inference session to run the model, with WebGL as the execution provider and enabling all graph optimizations
    session = await ort.InferenceSession.create(
      "./_next/static/chunks/pages/model.onnx",
      {
        executionProviders: ["webgl"],
        graphOptimizationLevel: "all",
      }
    );
    console.log("Inference session created");
  } catch (error) {
    console.error("Failed to load the model:", error);
  }
}

export async function runSHCModel(
  preprocessedData: any
): Promise<[any, number]> {
  if (!session) {
    throw new Error("Model not loaded");
  }

  // Run inference on the preprocessed data and get the results and the inference time
  const [results, inferenceTime] = await runInference(
    session,
    preprocessedData
  );
  return [results, inferenceTime];
}

// Function to run the inference
async function runInference(
  session: ort.InferenceSession,
  preprocessedData: any
): Promise<[any, number]> {
  // Start a timer
  const start = new Date();
  // Create an object to hold the input tensor
  const feeds: Record<string, ort.Tensor> = {};
  // Add the preprocessed data as the input tensor
  feeds[session.inputNames[0]] = preprocessedData;
  // Run the model with the input tensor and get the output data
  const outputData = await session.run(feeds);
  // Stop the timer and calculate the inference time
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  // Get the output data as a tensor
  const output = outputData[session.outputNames[0]];
  // Return the output data and the inference time
  return [output.data, inferenceTime];
}

export const calculateMovingAverage = (
  buffer: Float32Array[]
): Float32Array => {
  // Get the number of coefficients in the buffer
  const numCoefficients = buffer[0].length;
  // Create a new Float32Array to hold the averaged SHCs
  const averagedSHC = new Float32Array(numCoefficients);
  // For each coefficient in the buffer
  for (let i = 0; i < numCoefficients; i++) {
    // Initialize a sum to 0
    let sum = 0;
    // For each SHC in the buffer
    for (const shc of buffer) {
      // Add the value of the coefficient to the sum
      sum += shc[i];
    }
    // Calculate the average of the coefficient and store it in the averaged SHCs
    averagedSHC[i] = sum / buffer.length;
  }
  // Return the averaged SHCs
  return averagedSHC;
};
