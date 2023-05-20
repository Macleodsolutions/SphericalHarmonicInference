// Language: typescript
// Path: react-next\utils\predict.ts
import {getImageTensorFromPath} from './imageHelper';
import {loadSHCModel, runSHCModel} from './modelHelper';

let modelLoaded = false;

async function ensureModelLoaded(): Promise<void> {
    if (!modelLoaded) {
        await loadSHCModel();
        modelLoaded = true;
    }
}

export async function inference(path: Uint8ClampedArray, height: number, width: number): Promise<[any, number]> {
    // 0. Ensure the model is loaded
    await ensureModelLoaded();

    // 1. Convert image to tensor
    const imageTensor = await getImageTensorFromPath(path, [1, 3, height, width]);

    // 2. Run model
    const [predictions, inferenceTime] = await runSHCModel(imageTensor);

    // 3. Return predictions and the amount of time it took to inference.
    return [predictions, inferenceTime];
}
