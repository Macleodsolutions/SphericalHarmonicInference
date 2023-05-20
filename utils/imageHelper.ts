// Importing required modules
import {Tensor} from 'onnxruntime-web';
import {CanvasTexture, MeshBasicMaterial} from "three";

export async function getImageTensorFromPath(path: Uint8ClampedArray, dims: number[] = [1, 3, 128, 256]): Promise<Tensor> {
    // Convert image data to tensor and return
    return imageDataToTensor(path, dims);
}

export function drawImageWithFilters(
    ctx: CanvasRenderingContext2D,
    source: any,
    brightness: number,
    contrast: number,
    saturation: number,
    tintColor: string,
    WIDTH: number,
    HEIGHT: number
) {
    // Draw the source image on the canvas
    ctx.drawImage(source, 0, 0, WIDTH, HEIGHT);
    // Set the filter property for brightness, contrast, and saturation
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturation})`;
    // Set the fill color
    ctx.fillStyle = tintColor;
    // Set the global composite operation to 'multiply' to apply the fill color as a tint
    ctx.globalCompositeOperation = 'multiply';
    // Fill the canvas with the tint color
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    // Reset the global composite operation to 'source-over' (default value)
    ctx.globalCompositeOperation = "source-over";
}

export function createNewTexture(
    ctx: CanvasRenderingContext2D,
    skydomeMaterial: MeshBasicMaterial,
    newTexture: CanvasTexture,
    canvas: HTMLCanvasElement
) {
    // If newTexture is not provided, create a new texture from the canvas
    if (!newTexture) {
        newTexture = new CanvasTexture(canvas);
    }
    // Dispose of the old texture map from the material, if it exists
    skydomeMaterial.map?.dispose();
    // Set the new texture as the material's map
    skydomeMaterial.map = newTexture;
}

function imageDataToTensor(image: Uint8ClampedArray, dims: number[]): Tensor {
    // Get buffer data from image and create R, G, and B arrays.
    const imageBufferData = image;

    // These are common imagenet mean and standard deviation values for each color channel (R, G, B)
    // that will be used to normalize the pixel values
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Dimensions of the image
    const height = dims[2];
    const width = dims[3];
    const channels = dims[1];

    // Create a new Float32Array to hold the pixel data
    const float32Data = new Float32Array(channels * height * width);

    // Loop through the image buffer and extract the R, G, and B channels
    // Normalize the pixel values using the mean and standard deviation values
    for (let i = 0, j = 0; i < imageBufferData.length; i += 4, j++) {
        // R
        float32Data[j] = (imageBufferData[i + 2] / 255 - mean[0]) / std[0];
        // G
        float32Data[height * width + j] = (imageBufferData[i + 1] / 255 - mean[1]) / std[1];
        // B
        float32Data[2 * height * width + j] = (imageBufferData[i] / 255 - mean[2]) / std[2];
    }

    return new Tensor(float32Data, dims);
}

