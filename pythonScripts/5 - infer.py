import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import time
import onnxruntime

IMG_WIDTH, IMG_HEIGHT = 256, 128
BATCH_SIZE = 16
EXPORT_ONNX = True
TEST_IMG_PATH = "shanghai_bund.png"
MODEL_PATH = "./177000.torch"


# Load an image and apply transformations
def load_image(img_path, width, height):
    img = cv2.imread(img_path)
    transform = tf.Compose([
        tf.ToPILImage(),
        tf.Resize((height, width)),
        tf.ToTensor(),
        tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(img)


def main():
    # Load and preprocess the image
    image = load_image(TEST_IMG_PATH, IMG_WIDTH, IMG_HEIGHT)
    images = torch.zeros([1, 3, IMG_HEIGHT, IMG_WIDTH])
    images[0] = image

    # Set the device for computation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model and set it to evaluation mode
    net = torchvision.models.mobilenet_v2(pretrained=False, width_mult=1.0)
    net.classifier = torch.nn.Linear(in_features=1280, out_features=27, bias=True)
    net = net.to(device)
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()

    # Convert and save the model using TorchScript
    scripted_net = torch.jit.script(net)
    torch.jit.save(scripted_net, "model.pth")
    loaded_scripted_net = torch.jit.load("model.pth")
    img = torch.autograd.Variable(image, requires_grad=False).to(device).unsqueeze(0)

    # Export the model to ONNX and perform inference
    if EXPORT_ONNX:
        torch.onnx.export(net, images.to(device), "model.onnx", export_params=True, opset_version=13,
                          do_constant_folding=True)
        ort_session = onnxruntime.InferenceSession("model.onnx")
        img = torch.autograd.Variable(image, requires_grad=False).to('cpu').unsqueeze(0)
        input_data = img.numpy()
        start_time = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        prediction = torch.tensor(ort_outs[0])  # Convert the output to a PyTorch tensor
        end_time = time.time()  # Record the end time
        inference_time = end_time - start_time  # Calculate the inference time
        print("Inference time: {:.6f} seconds".format(inference_time))
        print(prediction.data.cpu().numpy())

    # Perform inference using the TorchScript model
    else:
        start_time = time.time()
        with torch.no_grad():
            prediction = loaded_scripted_net(img)

        inference_time = time.time() - start_time

        print("Inference time: {:.6f} seconds".format(inference_time))
        print(prediction.data.cpu().numpy())


if __name__ == "__main__":
    main()
