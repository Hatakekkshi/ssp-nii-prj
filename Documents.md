# Document project chatbot in the simulated world 

This project contains 2 part: `Uninty Application` and `Chatbot api system`. 

Unity application like as front end of system and  have some special action such as: take a photo, list object from camera, ... This module will have UI to receive input text from user, send message content to Chatbot system to get some action such as: show reply message, take a photo, ...

Chatbot api system is a resful api server run on Torch7 framework, and Lua programing. To run this server, need install torch7 framework and some package support build server. This environment will be provided via docker. ([docker](https://docs.docker.com/docker-for-windows/install/) is a software make virtual machine which is usually use on linux similar with virtualbox, vmware.. on window)


## 1. Run unity application 

- Open unity project (name: `New Unity Project - Copy (2)`) 
    - _Note: Because this project base on tutorial for beginer, so when re-open this project, code of this project will be reset (sorry for the inconvenience) ._
- Open scene `Done3` in folder `_Scenes` to recovery state scene.
- In the `Hierachy` window, open `Player` to change code controller. 
- Replace code from file `ThirdPersonUserControl.cs` to controll `Player` agent.
- Drag some object in `Hierachy` window to make relationship in code `ThirdPersonUserControl.cs`
    - public InputField messageUserInput;
    - public InputField messageBotRep;
    - public Image viewOfBot;
    - public Camera cameraBot;
    - public BoxCollider pushableBox;
    - public float timeShowMessagePlayer;
    - public Text logMessage;

## 2. Run server chatbot system

I need to clean and refactor code. Comming soon ... 