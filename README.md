# Navigation
Navigation assignment for the Deep Learning Nanodegree




## Dependencies

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning), in the `p1_navigation/` folder, and unzip (or decompress) the file.


## Files

1. Navigation_Solution.ipynb
Thy ipython notebook loads the environment and controls the training of the agent. 

2. agent.py
This file defines the Agent class and the Replay Buffer class.

3. model_dueling.py
The class DuelingQNetwork extends PyTorch.NN.Module and defines the Q-Network architecure.

4. Report.pdf
The project report

5. WeightsAtSolved.pth
Pickled file created using torch.save(). This stores the model weights at the point at which the episode is solved.

6. FinalWeight.pth
Pickled file created using torch.save(). This stores the model weights at the point at which training terminates(after 700 episodes).
