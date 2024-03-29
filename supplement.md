# Supplementary Instructions

## Image Descriptions

&emsp;&emsp;Image descriptions are used for the training of two components (textual dialogue response generator and Text-to-Image translator). For the textual dialogue response generator, accurate and detailed image descriptions facilitate the generation of more contextualized image descriptions in inference. For the Text-to-Image translator, high-quality image-text pairs are necessary to improve the performance of Stable Diffusion in fine-tuning stage. According to our preliminary experience, the performance will be degraded if model is trained with the image descriptions provided by PhotoChat. In addition, it is also necessary that we use [Gigapixel](https://www.topazlabs.com/gigapixel-ai) to further improve the quality of the images.

<table align="center">
  <tr>
    <td align="center"><b></b></td>
    <td align="center"><b>Images</b></td>
    <td align="center"><b>Descriptions</b></td>
  </tr>
  <tr>
    <td rowspan="2" align="center">(a)</td>
    <td rowspan="2" align="center"><img src="figs/supplement/description_1.jpg" height="120px" title="(a)"></td>
    <td>PhotoChat: The photo has your aunt Kailey. Objects in the photo: Woman, Face</td>
  </tr>
  <tr>
    <td>BLIP-2: a man giving a presentation in front of a group of people</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">(b)</td>
    <td rowspan="2" align="center"><img src="figs/supplement/description_2.jpg" height="120px" title="(b)"></td>
    <td>PhotoChat: Objects in the photo: Animal, Carnivore, Cat</td>
  </tr>
  <tr>
    <td>BLIP-2: a cat sitting on the floor with its paw on its head</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">(c)</td>
    <td rowspan="2" align="center"><img src="figs/supplement/description_3.jpg" height="120px" title="(c)"></td>
    <td>PhotoChat: Objects in the photo: Building, Chair, Drink</td>
  </tr>
  <tr>
    <td>BLIP-2: a bar with neon lights and a sign that says be amazing</td>
  </tr>
</table>

&emsp;&emsp;The table above shows several images from PhotoChat and demonstrates that the image descriptions generated by BLIP-2 are more accurate and detailed than those provided by PhotoChat. Therefore, we use these new image descriptions instead of those provided by PhotoChat. Specifically, new image descriptions can express: 
1. character's actions (e.g., "giving a presentation" in (a), "with its paw on its head" in (b)). 
2. the relative positional relationships between objects (e.g., "in front of" in (a)). 
3. scene information (e.g., "with neon lights" in (c)). 

## Prompts

&emsp;&emsp;The image descriptions generated by the textual dialogue generator are brief sentences. Although they are context-sensitive, a short sentence will not inspire Stable Diffusion to generate a perfect image. Thus we add some extra prompts following the generated image descriptions, that is, applying some prompt templates. For various categories of descriptions (e.g., buildings, landscapes, portraits), the use of corresponding prompt templates would significantly improve the quality of the images. However, our research topic is multimodal dialogue response generation, not just Text-to-Image. Specifically, the description is generated and used inside the model, from which we cannot decide which category it belongs to. Here we simply divide all descriptions into "Human" and "Others". The following table shows the prompt templates. 


<table align="center">
  <tr>
    <td align="center"><b>Categories</b></td>
    <td align="center"><b>Prompt Templates</b></td>
  </tr>
  <tr>
    <td align="center">Human</td>
    <td align="left"><font face="Times New Roman"><I>Description</I></font>, facing the camera, photograph, highly detailed face, depth of field, moody light, style by Yasmin Albatoul, Harry Fayt, centered, extremely detailed, Nikon D850, award winning photography</td>
  </tr>
  <tr>
    <td align="center">Others</td>
    <td align="left"><font face="Times New Roman"><I>Description</I></font>, depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.</td>
  </tr>
</table>

&emsp;&emsp;The goal of using both prompt templates and negative prompt is to stabilize the generated image in the style we desire and to improve the quality of the image. 

## Image Generation Results

&emsp;&emsp;As shown in the table below, we show images generated by Text-to-Image translator of both models, in order to better compare their performance in image generation. We can find that TIGER's Text-to-Image translator has better text comprehension and image generation capabilities, and the generated images have higher clarity and fidelity, and consistency with descriptions. Obviously, this will also significantly improve the dialogue experience.

<table align="center">
  <tr>
    <td align="center"><b>Image Description</b></td>
    <td align="center"><b>Divter</b></td>
    <td align="center"><b>TIGER</b></td>
  </tr>
  <tr>
    <td align="center">a small brown and white dog wearing a pink hoodie</td>
    <td align="center"><img src="figs/supplement/divter_image_1.jpg" height="120px"></td>
    <td align="center"><img src="figs/supplement/tiger_image_1.jpg" height="120px"></td>
  </tr>
  <tr>
    <td align="center">a muffin sitting on top of a plate on a counter</td>
    <td align="center"><img src="figs/supplement/divter_image_2.jpg" height="120px"></td>
    <td align="center"><img src="figs/supplement/tiger_image_2.jpg" height="120px"></td>
  </tr>
  <tr>
    <td align="center">a black leather jacket with the word van city on it</td>
    <td align="center"><img src="figs/supplement/divter_image_3.jpg" height="120px"></td>
    <td align="center"><img src="figs/supplement/tiger_image_3.jpg" height="120px"></td>
  </tr>
  <tr>
    <td align="center">a woman with glasses and blonde hair on a television screen</td>
    <td align="center"><img src="figs/supplement/divter_image_4.jpg" height="120px"></td>
    <td align="center"><img src="figs/supplement/tiger_image_4.jpg" height="120px"></td>
  </tr>
</table>

## Case Study

<table align="center">
  <tr>
    <td align="center"><b>Dialogue Context</b></td>
    <td align="center"><b>Response</b></td>
  </tr>
  <tr>
    <td rowspan="2" align="left">A: What have you been up to lately?<br>
                                 B: i am encouraging my friend now<br>
                                 A: That's good! I hope it is going well.<br>
                                 B: yeah somewhat<br>
                                 B: he is nervous<br>
                                 A: Why is he nervous?<br>
                                 B: he is going to perform in a stage<br>
                                 B: so he is in rehearsal ibfront of the mike<br>
                                 A: That sounds fun! Is he in a play or a concert?<br>
                                 B: speech competition<br>
                                 A: That would make me nervous too! But I bet your friend will do fine!<br>
                                 B: his name is Brixton<br>
                                 A: That is a nice name. I haven't heard it before.<br>
                                 B: yes<br>
                                 B: wanna see him standing infront of mike<br>
                                 A: Sure! I'd like that!<br>
    </td>
    <td align="left">PhotoChat:<br>
                     <center><img src="figs/supplement/case_PhotoChat.jpg" height="120px"></center><br>
                     Image Description: a man with long hair and a beard is holding a microphone
    </td>
  </tr>
  <tr>
    <td align="left">TIGER:<br>
                     <center><img src="figs/supplement/case_Tiger.png" height="120px" align="center"></center><br>
                     Image Description (Generated by Textual Dialogue Response Generator): a man in a black shirt standing in front of a microphone
    </td>
  </tr>
</table>

&emsp;&emsp;As shown in the table above, we illustrate a case where a response is generated given a dialogue context. The dialogue context is about `speech competition`. This example proves that TIGER can: 
1. accurately determine the timing of responding with an image; 
2. generate accurate and detailed image descriptions; 
3. generate a high-quality, high-resolution (768 * 768) image as a visual response.

# <img src="figs/tiger_logo.png" height="30px"> Discussion

## Dose the MLLMs like GPT-4 can solves this problem?

&emsp;&emsp;We believe that solving this problem with MLLMs (Multimodal Large Language Models) is very promising. However, the current mainstream MLLMs, such as GPT-4, LLaVA, only have the ability to perceive and understand images, and they can well solve the problems of Image Captioning, VQA, etc., but do not have the ability to generate images.

&emsp;&emsp;Our research focuses on `open-domain` `multimodal` dialogue generation `with image generation capability`.
- On the one hand, MLLMs with image generation capability are scarce. Based on our experience with this kind of MLLMs, such as [GILL](https://arxiv.org/abs/2305.17216), [Emu](https://arxiv.org/abs/2307.05222), the timing to respond with images, as well as the quality of the generated image and the relevance of the image with the conversational context are not satisfactory.
- On the other hand, as far as we know, there aren't any MLLMs focusing on multimodal open-domain conversation, and most of them are still focusing on VQA scenarios, including [MiniGPT-4](http://arxiv.org/abs/2304.10592), [LLaVA](http://arxiv.org/abs/2304.08485), [MultiModal-GPT](http://arxiv.org/abs/2305.04790), [InstructBLIP](http://arxiv.org/abs/2305.06500), [Qwen-VL](http://arxiv.org/abs/2308.12966), etc.  

&emsp;&emsp;Undeniably, applying MLLMs to our approach is a promising topic, and we will also refine it in our future work.

## Limitations

&emsp;&emsp;The Text-to-Image translator using Stable Diffusion faces a common issue with Diffusion Models, where it takes significantly longer to generate images compared to DALL-E and GAN-style models. In our case, setting the inference step to 20 results in a 5-second delay in generating image responses, negatively impacting the overall dialogue experience. We acknowledge that improving the inference speed of Diffusion Models and exploring GAN-style models that outperform Stable Diffusion are promising research directions for the future.