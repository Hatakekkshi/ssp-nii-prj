---
--- Created by phuongnm.
--- DateTime: 19/10/2017 06:58
---
require("Package")
print(processTaskRecieveMessage(JSON:encode({message = "hello"})))
print(processTaskRecieveMessage(JSON:encode({message = "what can u do"})))
print(processTaskRecieveMessage(JSON:encode({message = "what can u see"})))
print(processTaskRecieveMessage(JSON:encode({message = "thanks"})))
print(processTaskRecieveMessage(JSON:encode({message = "can u comeback position which u can see blue box"})))
print(processTaskRecieveMessage(JSON:encode({message = "comeback position which u can see blue box"})))
print(processTaskRecieveMessage(JSON:encode({message = "do u know the sky"})))

