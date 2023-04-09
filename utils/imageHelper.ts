import * as Jimp from 'jimp';
import { Tensor } from 'onnxruntime-web';

export async function getImageTensorFromPath(path: string, dims: number[] =  [1, 3, 512, 1024]): Promise<Tensor> {
  // 1. load the image  
  const image = await loadImageFromPath(path);
  // 2. convert to tensor
  const imageTensor = imageDataToTensor(image, dims);
  // 3. return the tensor
  return imageTensor;
}

async function loadImageFromPath(path: string): Promise<Jimp> {
  // Use Jimp to load the image and resize it.
  const imageData = await Jimp.default.read(path);

  return imageData;
}
function imageDataToTensor(image: Jimp, dims: number[]): Tensor {
  // 1. Get buffer data from image and create R, G, and B arrays.
  const imageBufferData = image.bitmap.data;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const height = dims[2];
  const width = dims[3];
  const channels = dims[1];

  const float32Data = new Float32Array(channels * height * width);

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0, j = 0; i < imageBufferData.length; i += 4, j++) {
    float32Data[j] = (imageBufferData[i + 2] / 255 - mean[0]) / std[0]; // R
    float32Data[height * width + j] = (imageBufferData[i + 1] / 255 - mean[1]) / std[1]; // G
    float32Data[2 * height * width + j] = (imageBufferData[i] / 255 - mean[2]) / std[2]; // B
  }

  // 3. Create the tensor object from onnxruntime-web.
  const tensor = new Tensor(float32Data, dims);
  return tensor;
}


