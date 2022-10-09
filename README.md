# Seamless-Cloning-in-Python
A python reproduction of Poisson Image Blend (Seamless Cloning) proposed by MSRUK on SIGGRAPH 2004.

The foreground is chosen as below:

<img width="146" alt="image" src="https://user-images.githubusercontent.com/20149275/194751247-c8408d2b-6f70-4c73-b4de-b39c22ff5d0a.png">

The background is chosen as below:

<img width="347" alt="image" src="https://user-images.githubusercontent.com/20149275/194751276-7bf0ebb6-343e-471f-9ca6-da77510937e3.png">

And the result is shown below:

<img width="347" alt="image" src="https://user-images.githubusercontent.com/20149275/194751369-80a40ac1-cabb-4117-a776-738450d958df.png">

Warning! Maths ahead!

You can find the source paper here:
[Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)

The basic idea of Poisson image blend is to address the seamless cloning problem. It is, in detail, to solve a sparse, relaxed, and positive definite linear system which can be expressed as Ax = b, in which A is a sparse Laplacian matrix corresponding to the unknown blended area, and b is the vector divergence of the source foreground image. 

泊松图像融合的提出是为了解决前景图像无缝融合进背景图像的问题。理论上，对这一问题的解法等同于求解一个稀疏、松弛且正定的线性系统。我们可以用线性表达式 Ax = b 来表达这样一个系统，其中 A 是前景区域内待求像素的拉普拉斯矩阵，而 b 是原前景图像的散度场。

More blend image ahead! You may easily figure out which part is the foreground in each of the following images:

<img width="491" alt="moon05_fuji075_blend" src="https://user-images.githubusercontent.com/20149275/194752421-77c06471-1d33-4398-80bc-506ec0c7a6b0.png">

<img width="452" alt="saturn01_nightsky05_blend" src="https://user-images.githubusercontent.com/20149275/194752433-c29d2771-9c6a-4122-a537-21a589e7052d.png">

<img width="362" alt="plane02_lake205_blend" src="https://user-images.githubusercontent.com/20149275/194752456-967cc5cf-879e-46c2-95f8-0f2cefaaf586.png">
