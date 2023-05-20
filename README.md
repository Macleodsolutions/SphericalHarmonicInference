<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />

<h3 align="center">![rain_demo.mp4](https://raw.githubusercontent.com/Macleodsolutions/SphericalHarmonicInference/main/screenshots/rain_demo.mp4)</h3>

<h3 align="center">Inferring spherical harmonic coefficients
</br> from non-hdr sources at interactive framerates
</br>using ONNX and WebGL
</h3>

  <p align="center">
    <br />
    <a href="https://macleodsolutions.github.io/SphericalHarmonicInference/">View Demo</a>
    ·
    <a href="https://github.com/Macleodsolutions/SphericalHarmonicInference/issues">Report Bug</a>
    ·
    <a href="https://github.com/Macleodsolutions/SphericalHarmonicInference/issues">Request Feature</a>
  </p>

### Built With

* [![Blender][Blender]][Blender-url]
* [![Python][Python]][Python-url]
* [![PyTorch][PyTorch]][PyTorch-url]

### Deployed With

* [![ONNX][ONNX]][ONNX-url]
* [![TypeScript][TypeScript]][TypeScript-url]
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![WebGL][WebGL]][WebGL-url]

<!-- ABOUT THE PROJECT -->
![Screenshot 2023-05-17 230935.png](https://raw.githubusercontent.com/Macleodsolutions/SphericalHarmonicInference/main/screenshots/screenshot_1.png)
![Screenshot 2023-05-17 215602.png](https://raw.githubusercontent.com/Macleodsolutions/SphericalHarmonicInference/main/screenshots/screenshot_2.png)

## Motivation

My primary motivation for these experiments was to deploy a client side pytorch solution from scratch. We've deployed
several production ready inference models, but this usually happens through a server-side api, and almost exclusively
using off-the-shelf frameworks tailored to our training sets.

What I was looking to do with this project was take a generic problem first and subsequently see how well I could
develop a pytorch solution from scratch. Bonus points for running a constant live inference process.

So our key acceptance criteria here is:

- Develop a pytorch model from scratch
- Deploy to client side browser using ONNX
- Low file size footprint
- (Bonus) Constant inference at interactive rates

## Ambient Spherical Harmonics

It was around this time I read some documentation that
referenced "[Ambient Spherical Harmonics](https://developers.google.com/ar/develop/lighting-estimation)" used in AR
development.
Using spherical harmonic coefficients to approximate hdr lighting isn't a new trick, but the interesting twist here is
that this process uses a machine learning model to infer the coefficient values on the fly using a live source:

![Ambient Spherical Harmonics](https://raw.githubusercontent.com/Macleodsolutions/SphericalHarmonicInference/main/screenshots/google_example.jpg)

This seemed like an ideal candidate for a trying out building a pytorch model, as hdr images are readily available from
multiple high quality sources, plus this meets the bonus criteria of running constant inference.

## Data Prep

This part was the most straight forward, PolyHaven provides an api for scraping their library of .exr images.</br>
After these are downloaded, we can do some data augmentation using Blender to re-render the .exrs with varying settings.

In this experiment I used the initial 600ish exrs from polyhaven and augmented them to roughly 6500.

Importantly, the Blender rendering is outputting both the .exr and .png versions of the scene. Then the resulting .exr
has its spherical harmonic coefficients extracted using cmgen from Google's Filament.</br>
At this point you can safely disregard the augmented .exr files. The only thing we need to train our model is the .png
and .txt files.

## Building the model

I knew my general model was going to use an rgb image and predict the resulting 27 signed floating point values, which
is pretty typical linear regression.
Initially I tried a simple MSE loss based training loop but this wasn't gaining any traction, so I started looking into
full network architectures.</br>

While I have some familiarity already with resnet, but the standard resnet50 clocks in at almost 300mb, making it
unsuitable for client side deployment.</br>
Rather then train first and then look at some type of pruning solution, I wanted to use a more compact architecture, and
indeed since the last time I used resnet there have been a myriad of smaller models aiming to equal and even eclipse
resnet on tasks using far fewer parameters.

After evaluating the model sizes and benchmarks, as well as taking into consideration both the architectures that
PyTorch already ships with support for, and onnx web deployment compatibility, I settled
on [MobileNet v2](https://paperswithcode.com/method/mobilenetv2) for its small footprint and easily adjustable "width"
option for number of trainable parameters. After deploying to .onnx, total file size was 9mb for the default width=1.0
and 3mb for width=0.5, both more then adequate for client side deployment.

## Hyperparameter tuning

All hyperparameter tuning was done automatically with [Optuna](https://optuna.org/). I have never needed to do much
hyperparameter tuning as most complete frameworks have some baselines already established, however Optuna made it a
breeze to discover the best Learning Rate and (Eventual) Weight Decay.

## Subsequent improvements

At this point the model was already performing moderately well in cursory testing, loss is measuring roughly 0.025 on
batch size 40 with 1024x512 resolution (The record would end up being 0.0078).

Previous frameworks I've used typically included a variable learning rate, so I figure mine being static was probably
not great. After checking out the docs I implemented
the [ReduceOnLRPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
scheduler.

While reading the documentation I came across the page for optimizers, which I've had a brief dabble in for accelerating
image to image translation using FP16 mixed precision. My takeaway from last time is
that [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) is essentially always a
safe choice, and is actually a correct implementation of the
original [Adam paper](https://arxiv.org/abs/1711.05101).</br>
Weight decay value was then provided by Optuna.

I was also initially converting inputs into tensors on the fly which is tremendously slow at scale. This was swapped out
for [DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) running on multiple threads.

## Deployment

Deployment with ONNX went off without a snag. My previous deployments of ONNX were to production environments using
webpack, and I always wasted a ton of time just getting the .wasm files for onnx to load correctly.

This time fortunately there was an incredibly
handy [template](https://github.com/microsoft/onnxruntime-nextjs-template). I hadn't used NextJS before but the setup
and editing ended up being super easy. ThreeJS for WebGL rendering.

## Conclusions

The total file size of deploying the width=0.5 model was 3mb and inferring a 256x128 image is accomplished in as little
as 2ms, adequate, if not exactly ideal for interactive framerates. An additional penalty is incurred reading the pixels
from the framebuffer each frame, which is where inferring at 1024x512 really suffers performance loss without
significant improvement in inference results.

To mitigate this, I've added frame skipping to only infer every N frames, as well as a moving color average to smooth
out the flickering between inconsistent inferences.

### Tensorboard Results(Click image for live stats):

[![Tensorboard Results](https://raw.githubusercontent.com/Macleodsolutions/SphericalHarmonicInference/main/screenshots/tensorboard_results.jpg)](https://tensorboard.dev/experiment/3kPMnGr0SnSopbq7L7Ubbw/#scalars&runSelectionState=eyJ3aWR0aF8wLjVfMjU2eDEyOF9iYXRjaDEiOmZhbHNlLCJ3aWR0aF8wLjVfMjU2eDEyOF9iYXRjaDgiOmZhbHNlfQ%3D%3D&_smoothingWeight=0.999)

<!-- GETTING STARTED -->

## Getting Started

### Scraping HDRI data from https://polyhaven.com/ api:

The scraping and data preparation are handled through two self-contained .blend files (https://www.blender.org/):

|                                                                                      |                                                                                                                                                                                                                                         |
|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [1 - scrapePolyHaven.blend](pythonScripts%2F1%20-%20scrapePolyHaven.blend)           |                                                                        Download and cache hdris and metadata, target resolution can be changed inside the script                                                                        |
| [2 - generateHdriVariants.blend](pythonScripts%2F2%20-%20generateHdriVariants.blend) | Use Blenders background world shader to deterministically generate .exrs/.pngs with varying rotation, flipping, and intensity,<br/>then calculate spherical harmonics coefficients using cmgen.exe (https://github.com/google/filament) |

### Training:

For this experiment I used a conda environment running on Ubuntu 20.04 under WSL2 on Windows 11.
</br>Anecdotally I'm a huge fan of the interoperability that's been provided by WSL, full speed linux GPU pass through
and easy filesystem interop, literally a dream dev environment.

1. Create a new conda environment:
   ```sh
   conda create -n SphericalHarmonicInference
   ```
2. Create a new conda environment:
   ```sh
   conda activate SphericalHarmonicInference
   ```
3. Clone the repo (or just the pythonScripts folder)
   ```sh
   git clone https://github.com/Macleodsolutions/SphericalHarmonicInference.git
   ```
4. Clone the repo (or just the pythonScripts folder)
   ```sh
   cd SphericalHarmonicInference/pythonScripts
   ```
5. Install required packages packages (using pip inside conda environment)
   ```sh
   pip install requirements
   ```

After install the required packages you can use the following scripts for training:

|                                                    |                                             |
|:---------------------------------------------------|:-------------------------------------------:|
| [3 - optuna.py](pythonScripts%2F3%20-%20optuna.py) |  Hyperparameter tuning https://optuna.org/  |
| [4 - train.py](pythonScripts%2F4%20-%20train.py)   |              Primary training               |
| [5 - infer.py](pythonScripts%2F5%20-%20infer.py)   | Test inference, export .pth and .onnx files |

### Deployment:

The NextJS template is ready to go, just swap in your model.onnx file and run "next dev" in your project terminal

<!-- ROADMAP -->

## Roadmap

- [x] Basic ONNX deployment
- [x] Add video source
- [x] Add camera source
- [x] Add adjustable parameters for gui
- [x] Add random panorama source
- [ ] Generate equirectangular image from cubemap camera render of live scene

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge

[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members

[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge

[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers

[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-url]: https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/macleodsolutions/

[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white

[Python-url]: https://www.python.org

[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white

[PyTorch-url]: https://pytorch.org/

[Blender]: https://img.shields.io/badge/blender-%23F5792A.svg?style=for-the-badge&logo=blender&logoColor=white

[Blender-url]: https://www.blender.org/

[TypeScript]: https://camo.githubusercontent.com/773cfd323f61dbc7301a98e28c69fbd0f27f491272f4acf48106936ca1d14c47/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d5479706553637269707426636f6c6f723d333137384336266c6f676f3d54797065536372697074266c6f676f436f6c6f723d464646464646266c6162656c3d

[TypeScript-url]: https://www.typescriptlang.org/

[ONNX]: https://camo.githubusercontent.com/34541c5f5752d606cf19100dad4540eebe6ed1eaf059950368c78c3cb22dde0d/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d4f4e4e5826636f6c6f723d303035434544266c6f676f3d4f4e4e58266c6f676f436f6c6f723d464646464646266c6162656c3d

[ONNX-url]: https://onnx.ai/

[WebGL]: https://camo.githubusercontent.com/a880d3a51c7d9da4896eabe1cc534f7d598da6d80bb5ed58bd92eb1063a937f6/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d576562474c26636f6c6f723d393930303030266c6f676f3d576562474c266c6f676f436f6c6f723d464646464646266c6162656c3d

[WebGL-url]: https://www.khronos.org/webgl/

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white

[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB

[React-url]: https://reactjs.org/