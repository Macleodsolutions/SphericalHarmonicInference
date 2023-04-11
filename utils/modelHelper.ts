import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

export async function loadSHCModel(): Promise<void> {
  try {
    session = await ort.InferenceSession.create('./_next/static/chunks/pages/SHC.onnx', {
      executionProviders: ['webgl'],
      graphOptimizationLevel: 'all',
    });
    console.log('Inference session created');
  } catch (error) {
    console.error('Failed to load the model:', error);
  }
}

export async function runSHCModel(preprocessedData: any): Promise<[any, number]> {
  if (!session) {
    throw new Error('Model not loaded');
  }

  const [results, inferenceTime] = await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

async function runInference(
    session: ort.InferenceSession,
    preprocessedData: any
): Promise<[any, number]> {
  const start = new Date();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  const outputData = await session.run(feeds);
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  const output = outputData[session.outputNames[0]];
  return [output.data, inferenceTime];
}