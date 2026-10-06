from egai.replay.world import ReplayWorld, ReplayNode
w=ReplayWorld("demo",[
 ReplayNode("root",0,{"problem":"x"},{"op":"inspect"},{"ok":True}),
 ReplayNode("fix",1,{"finding":"y"},{"op":"repair"},{"ok":True},"root")
])
cap=w.issue_capability(0)
print(w.view(cap)) # cannot see t=1
