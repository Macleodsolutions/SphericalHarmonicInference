import React, {useEffect, useRef} from 'react';
import {
    CanvasTexture,
    Mesh,
    MeshBasicMaterial,
    PerspectiveCamera,
    PointLight,
    Scene,
    SphereGeometry,
    WebGLRenderer,
} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {inference} from '../utils/predict';
import editShaderForSHC from '../utils/shaderHelper';
import {calculateMovingAverage} from '../utils/modelHelper';
import * as GUI from 'lil-gui';
import {createNewTexture, drawImageWithFilters} from "../utils/imageHelper";

// Define constants for the width and height of the canvas, as well as parameters for the skydome
const WIDTH = 256;
const HEIGHT = 128;
const SKYDOME_SCALE_FACTOR = -1;
const SKYDOME_RADIUS = 500;
const SKYDOME_WIDTH_SEGMENTS = 60;
const SKYDOME_HEIGHT_SEGMENTS = 40;

// Define the paths to the image and video sources
const IMAGE_SRC = './_next/static/chunks/pages/shanghai_bund.png';
//const RANDOM_IMAGE_SRC = 'https://source.unsplash.com/random/?Panorama&1';
const VIDEO_SRC = './_next/static/chunks/pages/pexels-anna-hinckel-6128683-1920x1080-50fps.mp4';

// Define an interface for the properties of the ThreeScene component
interface ThreeSceneProps {
    width: number;
    height: number;
}

// Instantiate a new Map object to cache the spherical harmonic coefficients for each frame
const SHCache = new Map();

// Declare global variables for orbit controls, materials, canvas and its context, scene, caching, and source flags
let controls: OrbitControls;
let sphereMaterial: MeshBasicMaterial;
let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let skydomeMaterial: MeshBasicMaterial;
const scene = new Scene();

// Declare and initialize variables for the current frame, frame count, and renderer
let currentFrame: string = '0';
let frameCount = 0;
let renderer: WebGLRenderer;

// Declare and initialize boolean flags for caching and source types
let isCaching = false;
let isCameraSource = false;
let isVideoSource = false;
let isImageSource = true;

// Declare and initialize variables for image filters and frame parameters
let brightness = 99;
let contrast = 125;
let saturation = 1;
let tintColor = '#ffffff';
let frameSkip = 0;
let frameSmoothing = 5;

const ThreeScene: React.FC<ThreeSceneProps> = ({width, height}) => {
    const containerRef = useRef<HTMLDivElement>(null);

    // Create a reference to buffer for Spherical Harmonic Coefficients (SHC)
    const SHCBufferRef = useRef<Float32Array[]>([]);

    const camera = new PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Create a reference to a GUI object using useRef React hook
    const guiRef = useRef<GUI.GUI | null>(null);

    // Create a reference to an object that holds the type of source used for the image
    const sourceType = useRef({type: 'image'});
    let source: any;
    let newTexture: CanvasTexture;

    const setupSource = (updateTexture: () => Promise<void>, type: string) => {
        // If the type is 'camera', access the user's webcam and set it as the source
        if (type === 'camera') {
            source = document.createElement('video');
            navigator.mediaDevices
                .getUserMedia({video: true})
                .then((stream) => {
                    source.srcObject = stream;
                    source.play();
                })
                .catch((error) => {
                    console.error('Error accessing webcam:', error);
                });
            // If the type is 'video', create a video element and set its source to the video file, also set it to loop and mute, and start playing it
        } else if (type === 'video') {
            source = document.createElement('video');
            source.src = VIDEO_SRC;
            source.loop = true;
            source.muted = true;
            source.play();
            source.addEventListener('timeupdate', () => {
                if (source.currentTime) {
                    currentFrame = source.currentTime.toFixed(1);
                }
            });
            // If the type is 'image', call the inferImage function with the updateTexture function as the parameter
        } else if (type === 'image') {
            inferImage(updateTexture);
        }
    };

    // Define a function to load an image and update its texture, this function accepts a function to update the texture as a parameter
    function inferImage(updateTexture: () => Promise<void>) {
        source = document.createElement('image');

        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.src = IMAGE_SRC;

        // Set an onload event listener to the Image object which will call the updateTexture function when the image is fully loaded
        img.onload = async function () {
            await updateTexture();
        }
        source = img;
    }

    function sceneSetup() {

        controls = new OrbitControls(camera, renderer.domElement);

        const sphereGeometry = new SphereGeometry(1, 32, 32);
        sphereMaterial = new MeshBasicMaterial({color: 0xffffff});

        // Edit the material's shader for Spherical Harmonic Coefficients
        editShaderForSHC(sphereMaterial);

        const sphere = new Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);

        const pointLight = new PointLight(0xffffff, 1);
        pointLight.position.set(10, 10, 10);
        scene.add(pointLight);

        canvas = document.createElement('canvas');
        canvas.width = WIDTH;
        canvas.height = HEIGHT;

        ctx = canvas.getContext('2d', {willReadFrequently: true})!;

        const skydomeGeometry = new SphereGeometry(SKYDOME_RADIUS, SKYDOME_WIDTH_SEGMENTS, SKYDOME_HEIGHT_SEGMENTS).scale(SKYDOME_SCALE_FACTOR, 1, 1);

        // Create a basic material for the skydome and set its map to a new CanvasTexture created from the canvas
        skydomeMaterial = new MeshBasicMaterial({map: new CanvasTexture(canvas)});
        const skydome = new Mesh(skydomeGeometry, skydomeMaterial);
        scene.add(skydome);

        // Rotate the skydome mesh around the Y-axis by 1.57 radians (or 90 degrees)
        skydome.rotateY(1.57);
    }

    function setupGui(updateTexture: () => Promise<void>) {
        // Initialize the GUI and store its reference
        guiRef.current = new GUI.GUI();

        const guiSettings = [
            // Each setting is an object that contains the name, initial value, range (if applicable), and a callback function for when the value is changed
            {name: 'brightness', value: brightness, range: [0, 200], callback: (value: number) => brightness = value},
            {name: 'contrast', value: contrast, range: [0, 200], callback: (value: number) => contrast = value},
            {name: 'saturation', value: saturation, range: [0, 2], callback: (value: number) => saturation = value},
            {name: 'Tint', value: tintColor, type: 'color', callback: (value: string) => tintColor = value},
            {name: 'Caching', value: isCaching, type: 'checkbox', callback: (value: boolean) => isCaching = value},
            {name: 'Frame Skip', value: frameSkip, range: [0, 60], callback: (value: number) => frameSkip = value},
            {
                name: 'Frame Smoothing',
                value: frameSmoothing,
                range: [1, 60],
                callback: (value: number) => frameSmoothing = value
            },
        ];

        // Iterate over the settings array
        for (const setting of guiSettings) {
            // If the setting type is color, add a color controller to the GUI
            if (setting.type === 'color') {
                guiRef.current.addColor({[setting.name]: setting.value}, setting.name).onChange(setting.callback);
            }
            // If the setting type is checkbox, add a checkbox controller to the GUI
            else if (setting.type === 'checkbox') {
                guiRef.current.add({[setting.name]: setting.value}, setting.name).onChange(setting.callback);
            }
            // For other types, add a slider to the GUI
            else {
                guiRef.current.add({[setting.name]: setting.value}, setting.name, ...setting.range!).name(setting.name).onChange((value: number | string | boolean) => {
                    // Call the callback function when the slider value is changed
                    // @ts-ignore
                    setting.callback(value);

                    // If the source is an image, infer the image and update the texture
                    if (isImageSource) {
                        inferImage(updateTexture);
                    }
                });
            }
        }
        // Create a folder named 'Source Type' in the GUI
        const folder = guiRef.current.addFolder('Source Type');

        // Add a controller to choose the source type in the folder
        folder
            .add(sourceType.current, 'type', ['camera', 'video', 'image'])
            .name('Source')
            .onChange((value: string) => {
                // Update the source type flags and setup the source when the source type is changed
                isCameraSource = value === 'camera';
                isVideoSource = value === 'video';
                isImageSource = value === 'image';
                setupSource(updateTexture, value);
            });

        // Open the folder by default
        folder.open();
    }

    // React hook that runs once after the initial render
    useEffect(() => {
        if (!containerRef.current) {
            return;
        }

        renderer = new WebGLRenderer();
        containerRef.current.appendChild(renderer.domElement);

        const runInference = async (ctx: CanvasRenderingContext2D) => {
            // Get image data from the context
            const myImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            // Run inference on the image data
            const [inferenceResult] = await inference(myImageData.data, HEIGHT, WIDTH);

            // Return the result as a Float32Array
            return new Float32Array(inferenceResult);
        };

        const inferImageData = async () => {
            if (!ctx) return new Float32Array();
            drawImageWithFilters(ctx, source, brightness, contrast, saturation, tintColor, WIDTH, HEIGHT);
            createNewTexture(ctx, skydomeMaterial, newTexture, canvas);
            return await runInference(ctx);
        };


        const updateTexture = async () => {
            if (ctx) {
                // If caching is enabled and the current frame is in the cache
                if (isCaching && SHCache.has(currentFrame)) {
                    // Draw the source image on the canvas
                    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

                    // Dispose of the current material map
                    skydomeMaterial.map?.dispose();

                    // Set the new material map from the canvas
                    skydomeMaterial.map = new CanvasTexture(canvas);

                    // Set the Spherical Harmonic Coefficients from the cache
                    sphereMaterial.userData.SphericalHarmonicCoefficients.value = SHCache.get(currentFrame);
                }
                // If the source is a video
                else if (isVideoSource) {
                    // Push the inferred image data to the buffer
                    SHCBufferRef.current.push(await inferImageData());

                    // If the buffer is larger than the frame smoothing value, remove the oldest frame
                    if (SHCBufferRef.current.length > frameSmoothing) SHCBufferRef.current.shift();

                    // Calculate the moving average of the buffer
                    const averagedSHC = calculateMovingAverage(SHCBufferRef.current);

                    // Set the Spherical Harmonic Coefficients to the moving average
                    sphereMaterial.userData.SphericalHarmonicCoefficients.value = averagedSHC;

                    // If caching is enabled, store the moving average in the cache
                    if (isCaching) SHCache.set(currentFrame, averagedSHC);
                }
                // For other sources
                else if (isCameraSource || isImageSource) {
                    // Set the Spherical Harmonic Coefficients to the inferred image data
                    sphereMaterial.userData.SphericalHarmonicCoefficients.value = await inferImageData();
                }
            }
        };

        const animate = () => {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
            controls.update();

            // Update the skydome for every frame regardless of frame skipping
            if (!isImageSource) {
                drawImageWithFilters(ctx, source, brightness, contrast, saturation, tintColor, WIDTH, HEIGHT);
                createNewTexture(ctx, skydomeMaterial, newTexture, canvas);
            }

            // Perform inference only when frame count is greater than or equal to the frame skip value (TODO: double updating skydome, minimal effect on fps, probably not worth fixing
            if (!isImageSource && (++frameCount >= frameSkip)) {
                updateTexture();
                frameCount = 0;
            }
        };

        // START
        sceneSetup();
        setupGui(updateTexture);
        inferImage(updateTexture);
        animate();

        // Return a cleanup function that runs when the component is unmounted
        return () => {
            containerRef.current?.removeChild(renderer.domElement);
            controls.dispose();
            renderer.dispose();
            guiRef.current?.destroy();
        };
    }, []);

    // React hook that runs when the width or height changes
    useEffect(() => {
        if (!containerRef.current || !renderer) return;
        renderer.setSize(width, height);
    }, [width, height]);

    return <div ref={containerRef}/>;
};

export default ThreeScene;




