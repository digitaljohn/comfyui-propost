# ComfyUI Pro Post

A set of custom ComfyUI nodes for performing basic post-processing effects. These effects can help to take the edge off AI imagery and make them feel more natural. We only have two nodes at the moment, but we plan to add more over time.

![ComfyUI Screenshot using Pro Post](./examples/propost.jpg)

## Installation

- Navigate to the `/ComfyUI/custom_nodes/` folder
- Run `git clone https://github.com/digitaljohn/comfyui-propost.git`
- Restart ComfyUI

## Nodes

### Film Grain

A film grain effect with many parameters to control the look of the grain. It can create different noise types and patterns, and it can be used to create a variety of film grain looks.

![Pro Post Film Grain Example](./examples/propost-filmgrain.jpg)

| Parameter   | Default   | Type    | Description                                                                             |
|-------------|-----------|---------|-----------------------------------------------------------------------------------------|
| gray_scale  | `false`   | Boolean | Enables grayscale mode. If true, the output will be in grayscale.                       |
| grain_type  | `Fine`    | String  | Sets the grain type. Values can be Fine, Fine Simple, Coarse, or Coarser.               |
| grain_sat   | `0.5`     | Float   | Grain color saturation, with a range of 0.0 to 1.0.                                     |
| grain_power | `0.7`     | Float   | Overall intensity of the grain effect.                                                  |
| shadows     | `0.2`     | Float   | Intensity of grain in the shadows.                                                      |
| highs       | `0.2`     | Float   | Intensity of grain in the highlights.                                                   |
| scale       | `1.0`     | Float   | Image scaling ratio. Scales the image before applying grain and scales back afterwards. |
| sharpen     | `0`       | Integer | Number of sharpening passes.                                                            |
| src_gamma   | `1.0`     | Float   | Gamma compensation applied to the input.                                                |
| seed        | `1`       | Integer | Seed for the grain random generator.                                                    |

> Note: This code is a direct port/lift of the versatile `Filmgrainer` library available here: https://github.com/larspontoppidan/filmgrainer


### Vignette

A simple vignette effect that darkens the edges of the screen. It supports very subtle vignettes, as well as more pronounced ones.

![Pro Post Film Grain Example](./examples/propost-vignette.jpg)

| Parameter   | Default   | Type    | Description                                                        |
|-------------|-----------|---------|--------------------------------------------------------------------|
| intensity   | `1.0`     | Float   | The intensity of the vignette effect, with a range of 0.0 to 10.0. |


### Radial Blur

This filter allows you to blur the edges of the image. It has a few different options that allows you to accomplish a variety of effects.

![Pro Post Radial Blur Example](./examples/propost-radialblur.jpg)


| Parameter            | Default   | Type    | Description                                                                   |
|----------------------|-----------|---------|-------------------------------------------------------------------------------|
| edge_blur_strength   | `64.0`    | Float   | The intensity of the blur at the edges, with a range of 0.0 to 200.0.         |
| center_focus_weight  | `1.0`     | Float   | How focused the blur is. A smaller value pinches the blur towards the center. |
| steps                | `5`       | Integer | The number of steps to use when blurring the image. Higher numbers are slower.|

> Note: Using steps set to `1` can create some dreamy effects as seen below with a high edge_blur_strength.

![Pro Post Dreamy Radial Blur Example](./examples/propost-radialblur-dreamy.jpg)


## Putting it all together

Obviously due to the nature of ComfyUI you can compose these effects together. Below is an example of a strong vignette along with a coarse film grain effect.

![Pro Post Film Grain Example](./examples/propost-compound.jpg)

## Example

Check out the sample workflow that was used to generate these sample images here: [Pro Post Example Workflow](./examples/propost.json). Have fun!
