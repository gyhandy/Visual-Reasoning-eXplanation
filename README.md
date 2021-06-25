# Visual-Reasoning-eXplanation

[CVPR 2021 A Peek Into the Reasoning of Neural Networks: Interpreting with Structural Visual Concepts] 

### [Project Page](http://ilab.usc.edu/andy/vrx) | [Video](https://youtu.be/ZzkpUrK-cRA) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_A_Peek_Into_the_Reasoning_of_Neural_Networks_Interpreting_With_CVPR_2021_paper.pdf)
<div align="center">
    <img src="./docs/Fig-1.png" alt="Editor" width="500">
</div>



**Figure:** *An example result with the proposed VRX. To explain the prediction (i.e., fire engine and not alternatives like ambulance), VRX provides both visual and structural clues.*

> **A Peek Into the Reasoning of Neural Networks: Interpreting with Structural Visual Concepts** <br>
> Yunhao Ge,  Yao Xiao, Zhi Xu, Meng Zheng, Srikrishna Karanam, Terrence Chen, Laurent Itti, Ziyan Wu <br>
> *IEEE/ CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2021*


We considered the challenging problem of interpreting the reasoning logic of a neural network decision. We propose a novel framework to interpret neural networks which extracts relevant class-specific visual concepts and organizes them using structural concepts graphs based on pairwise concept relationships. By means of knowledge distillation, we show VRX can take a step towards mimicking the reasoning process of NNs and provide logical, concept-level explanations for final model decisions. With extensive experiments, we empirically show VRX can meaningfully answer “why” and “why not” questions about the prediction, providing easy-to-understand insights about the reasoning process. We also show that these insights can potentially provide guidance on improving NN’s performance.

<div align="center">
    <img src="./docs/Fig-2.png" alt="Editor" width="700">
</div>
**Figure:** *Examples of representing images as structural concept graph.*

<div align="center">
    <img src="./docs/Fig-3.png" alt="Editor" width="700">
</div>
**Figure:** *Pipeline for Visual Reasoning Explanation framework.*


Please see here for a re-implementation from sssufmug [[Code](https://github.com/sssufmug/visual-reasoning-explanation)]

**We are actively organizing the code and will publish a full version in few days; please keep an eye on our Github. Thanks! **   

